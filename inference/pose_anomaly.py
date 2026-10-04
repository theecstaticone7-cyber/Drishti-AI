"""
Pose-based anomaly scoring for Drishti AI.

The model is an LSTM autoencoder trained ONLY on normal human motion
(ShanghaiTech Campus training set, which contains no anomalies).
At inference time it tries to reconstruct a short window of a person's
skeleton; motion it has never seen (running, fighting, falling, throwing...)
reconstructs badly, so reconstruction error is the anomaly score.

This file is shared by the training notebook and the Streamlit app so that
both use exactly the same keypoint layout and normalisation.
"""

from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Keypoint layout
# --------------------------------------------------------------------------
# ShanghaiTech poses (AlphaPose) are COCO-17. MediaPipe gives 33 landmarks;
# these are the MediaPipe indices of the 17 COCO joints, in COCO order:
# nose, L/R eye, L/R ear, L/R shoulder, L/R elbow, L/R wrist,
# L/R hip, L/R knee, L/R ankle
MEDIAPIPE_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# COCO-17 -> 18 joints (adds a neck = mean of shoulders) and reorders,
# matching the layout used by the STG-NF ShanghaiTech pose release.
_COCO18_ORDER = np.array([0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3])

NUM_JOINTS = 18
SEG_LEN = 24              # frames per window
VID_RES = (856.0, 480.0)  # ShanghaiTech resolution, used as a fixed pixel scale


def coco17_to_coco18(kps):
    """kps: (..., 17, C) -> (..., 18, C)"""
    kps = np.asarray(kps, dtype=np.float32)
    neck = 0.5 * (kps[..., 5, :] + kps[..., 6, :])
    kps = np.concatenate([kps, neck[..., None, :]], axis=-2)
    return kps[..., _COCO18_ORDER, :]


def mediapipe_to_coco17_pixels(landmarks, box_xyxy):
    """
    Convert MediaPipe landmarks computed on a person crop into COCO-17
    pixel coordinates in the full frame.

    landmarks: results.pose_landmarks.landmark (33 items, x/y normalised to the crop)
    box_xyxy:  (x1, y1, x2, y2) of the crop in the full frame
    returns:   (17, 2) float32 array
    """
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    out = np.empty((17, 2), dtype=np.float32)
    for i, mp_idx in enumerate(MEDIAPIPE_TO_COCO17):
        lm = landmarks[mp_idx]
        out[i, 0] = x1 + lm.x * w
        out[i, 1] = y1 + lm.y * h
    return out


def normalize_segments(segs_xy):
    """
    Normalise pose windows the same way STG-NF does for ShanghaiTech:
    scale by a fixed resolution, centre each window on its mean position,
    and divide by the std of its y coordinates. This removes where the
    person is in the frame and how big they look, keeping only body shape
    and motion.

    segs_xy: (N, T, V, 2) pixel coordinates
    returns: (N, T, V, 2) float32
    """
    x = np.asarray(segs_xy, dtype=np.float32) / np.array(VID_RES, dtype=np.float32)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std_y = x[..., 1].std(axis=(1, 2))[:, None, None, None]
    std_y = np.maximum(std_y, 1e-6)
    return (x - mean) / std_y


def speed_score(segs_xy):
    """
    Rule-based motion score: average joint speed per frame, divided by body
    size (std of y). Fast motion (running, cycling) gives high values.
    segs_xy: (N, T, V, 2) pixel coordinates -> (N,)
    """
    segs_xy = np.asarray(segs_xy, dtype=np.float32)
    height = segs_xy[..., 1].std(axis=(1, 2)) + 1e-6
    speed = np.linalg.norm(np.diff(segs_xy, axis=1), axis=-1).mean(axis=(1, 2))
    return speed / height


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class PoseLSTMAutoencoder(nn.Module):
    """Seq2seq LSTM autoencoder over (T, 18*2) pose windows."""

    def __init__(self, num_joints=NUM_JOINTS, hidden_dim=128, latent_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        in_dim = num_joints * 2
        self.encoder = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        # x: (B, T, V, 2)
        B, T, V, C = x.shape
        flat = x.reshape(B, T, V * C)
        _, (h, _) = self.encoder(flat)
        z = self.to_latent(h[-1])                      # (B, latent)
        dec_in = self.from_latent(z).unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_in)
        recon = self.out(dec_out).reshape(B, T, V, C)
        return recon

    @torch.no_grad()
    def anomaly_score(self, x):
        """Mean squared reconstruction error per window. Higher = more anomalous."""
        recon = self(x)
        return ((recon - x) ** 2).mean(dim=(1, 2, 3))


class PoseTransformerAutoencoder(nn.Module):
    """
    Transformer autoencoder over (T, 18*2) pose windows.

    Encoder: self-attention across the 24 frames, then mean-pooled into a small
    latent vector. That bottleneck stops the model from simply copying its
    input. Decoder: the latent vector plus learned frame positions, refined by
    more self-attention layers and projected back to joint coordinates.
    """

    def __init__(self, num_joints=NUM_JOINTS, seg_len=SEG_LEN, d_model=128, latent_dim=64,
                 num_layers=3, nhead=4, dropout=0.1):
        super().__init__()
        in_dim = num_joints * 2
        self.inp = nn.Linear(in_dim, d_model)
        self.enc_pos = nn.Parameter(torch.zeros(1, seg_len, d_model))
        self.dec_pos = nn.Parameter(torch.zeros(1, seg_len, d_model))
        nn.init.trunc_normal_(self.enc_pos, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)

        def stack(n):
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                               dropout=dropout, batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, n, enable_nested_tensor=False)

        self.encoder = stack(num_layers)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.decoder = stack(num_layers)
        self.out = nn.Linear(d_model, in_dim)

    def forward(self, x):
        B, T, V, C = x.shape
        h = self.inp(x.reshape(B, T, V * C)) + self.enc_pos[:, :T]
        z = self.to_latent(self.encoder(h).mean(dim=1))           # (B, latent)
        d = self.from_latent(z).unsqueeze(1) + self.dec_pos[:, :T]
        recon = self.out(self.decoder(d)).reshape(B, T, V, C)
        return recon

    @torch.no_grad()
    def anomaly_score(self, x):
        recon = self(x)
        return ((recon - x) ** 2).mean(dim=(1, 2, 3))


ARCHS = {"lstm": PoseLSTMAutoencoder, "transformer": PoseTransformerAutoencoder}


def build_model(cfg):
    """Build a model from a checkpoint/training config dict."""
    arch = cfg.get("arch", "lstm")
    if arch == "lstm":
        return PoseLSTMAutoencoder(hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
                                   num_layers=cfg["num_layers"])
    return PoseTransformerAutoencoder(seg_len=cfg.get("seg_len", SEG_LEN), d_model=cfg["hidden_dim"],
                                      latent_dim=cfg["latent_dim"], num_layers=cfg["num_layers"])


# --------------------------------------------------------------------------
# Online scorer used by the app
# --------------------------------------------------------------------------
class PoseAnomalyScorer:
    """
    Keeps a sliding window of poses per tracked person and scores it.

    Checkpoint format (written by the training notebook):
        {
          "state_dict": ...,
          "config": {"arch": "lstm"|"transformer", "hidden_dim": .., "latent_dim": .., "num_layers": .., "seg_len": ..},
          "threshold": float,   # autoencoder-only threshold, calibrated on held-out normal data
          "fusion": {"err_mean", "err_std", "speed_mean", "speed_std", "threshold"},  # optional
          "metrics": {...}      # test results, for reference
        }

    If "fusion" is present, the score is z(reconstruction error) + z(speed),
    standardised with statistics from held-out normal training clips.
    Otherwise it is the raw reconstruction error.
    """

    def __init__(self, ckpt_path, device="cpu", max_missing=10):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.seg_len = cfg.get("seg_len", SEG_LEN)
        self.arch = cfg.get("arch", "lstm")
        self.model = build_model(cfg).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.fusion = ckpt.get("fusion")
        self.threshold = float(self.fusion["threshold"] if self.fusion else ckpt["threshold"])
        self.metrics = ckpt.get("metrics", {})
        self.device = device
        self.max_missing = max_missing
        self._buffers = defaultdict(lambda: deque(maxlen=self.seg_len))
        self._last_seen = {}
        self._frame_idx = 0

    def step(self):
        """Call once per processed frame; drops tracks that disappeared."""
        self._frame_idx += 1
        stale = [t for t, f in self._last_seen.items() if self._frame_idx - f > self.max_missing]
        for t in stale:
            self._buffers.pop(t, None)
            self._last_seen.pop(t, None)

    def update(self, track_id, kp17_xy):
        """
        Add one COCO-17 pixel pose for a track. Returns the anomaly score
        once the window is full, otherwise None.
        """
        self._last_seen[track_id] = self._frame_idx
        buf = self._buffers[track_id]
        buf.append(coco17_to_coco18(kp17_xy))
        if len(buf) < self.seg_len:
            return None
        raw = np.stack(buf)[None]                              # (1, T, 18, 2) pixels
        x = torch.from_numpy(normalize_segments(raw)).to(self.device)
        err = float(self.model.anomaly_score(x)[0])
        if not self.fusion:
            return err
        f = self.fusion
        spd = float(speed_score(raw)[0])
        return (err - f["err_mean"]) / f["err_std"] + (spd - f["speed_mean"]) / f["speed_std"]

    def is_anomalous(self, score):
        return score is not None and score > self.threshold

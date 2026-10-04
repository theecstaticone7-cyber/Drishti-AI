"""
Train and evaluate the Drishti AI pose anomaly model on ShanghaiTech Campus.

Protocol (standard for ShanghaiTech, same as STG-NF, ICCV 2023):
  * train only on the training split, which contains normal behaviour only
  * score every test frame, smooth per clip, report frame-level ROC-AUC

Pose data, ground truth and the scoring code come from the STG-NF repository
(https://github.com/orhir/STG-NF), so our numbers are directly comparable
with theirs. Run this from the Colab notebook in notebooks/, or locally:

    python training/train_eval_shanghaitech.py --stgnf_dir /path/to/STG-NF

Outputs (in --out_dir):
    pose_ae_shanghaitech.pt   checkpoint used by inference/app.py
    results.json              all metrics
    roc_curve.png             ROC curves for every method
"""

import argparse
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "inference"))
from pose_anomaly import SEG_LEN, build_model, normalize_segments  # noqa: E402


# --------------------------------------------------------------------------
# STG-NF data + scoring helpers
# --------------------------------------------------------------------------
def import_stgnf(stgnf_dir):
    """Import STG-NF's loader/scorer, patching two incompatibilities with new libraries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hasattr(np, "int"):
        np.int = int  # removed in NumPy 1.24, still used by STG-NF
    orig_style = plt.style.use
    plt.style.use = lambda *a, **k: None  # 'seaborn-ticks' style no longer exists
    sys.path.insert(0, stgnf_dir)
    try:
        from dataset import gen_dataset
        from utils import scoring_utils
    finally:
        plt.style.use = orig_style
    return gen_dataset, scoring_utils


def load_split(gen_dataset, pose_dir, seg_len, seg_stride):
    # STG-NF's gen_dataset only works with ret_keys=True and ret_global_data=True
    segs, meta, _, _, _, _ = gen_dataset(pose_dir, seg_len=seg_len, seg_stride=seg_stride,
                                         ret_keys=True, ret_global_data=True, dataset="ShanghaiTech")
    # segs: (N, 3, T, 18) -> xy only, (N, T, 18, 2)
    xy = np.transpose(segs[:, :2], (0, 2, 3, 1))
    return xy, np.array(meta)


def frame_level_eval(scoring_utils, normality, meta, seg_len):
    """Per-frame gt and smoothed normality scores using STG-NF's protocol (higher = more normal)."""
    args = Namespace(dataset="ShanghaiTech", seg_len=seg_len)
    gt_arr, scores_arr = scoring_utils.get_dataset_scores(np.asarray(normality, dtype=np.float64),
                                                          np.asarray(meta), args=args)
    scores_arr = scoring_utils.smooth_scores(scores_arr)
    gt = np.concatenate(gt_arr)          # 1 = normal, 0 = anomalous
    scores = np.concatenate(scores_arr)
    finite = np.isfinite(scores)
    scores[~finite & (scores > 0)] = scores[finite].max()
    scores[~finite & (scores < 0)] = scores[finite].min()
    return gt, scores


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------
def speed_baseline(xy):
    """Rule-based baseline: average joint speed, scaled by body size. Fast motion = anomalous."""
    height = xy[..., 1].std(axis=(1, 2)) + 1e-6
    speed = np.linalg.norm(np.diff(xy, axis=1), axis=-1).mean(axis=(1, 2))
    return speed / height


def train_autoencoder(train_x, val_x, cfg, device, log):
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    train_t = torch.from_numpy(train_x)
    val_t = torch.from_numpy(val_x).to(device)
    best_val, best_state = float("inf"), None
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(train_t))
        total = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            batch = train_t[perm[i:i + cfg["batch_size"]]].to(device)
            loss = ((model(batch) - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch)
        sched.step()
        model.eval()
        val_loss = batched_scores(model, val_t, cfg["batch_size"]).mean()
        log(f"epoch {epoch + 1:2d}/{cfg['epochs']}  train {total / len(train_t):.4f}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def batched_scores(model, x, batch_size):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            b = x[i:i + batch_size]
            if not torch.is_tensor(b):
                b = torch.from_numpy(b)
            out.append(model.anomaly_score(b.to(next(model.parameters()).device)).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stgnf_dir", required=True, help="Cloned STG-NF repo with data/ShanghaiTech inside")
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "inference", "models"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--archs", default="lstm,transformer", help="Comma-separated models to train and compare")
    p.add_argument("--deploy", default="transformer", help="Which trained model the app loads")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--train_stride", type=int, default=6)
    p.add_argument("--val_frac", type=float, default=0.1, help="Fraction of training clips held out for calibration")
    p.add_argument("--threshold_pct", type=float, default=99.0, help="Percentile of held-out normal errors used as alert threshold")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda m: print(m, flush=True)  # noqa: E731

    stgnf_dir = os.path.abspath(args.stgnf_dir)
    gen_dataset, scoring_utils = import_stgnf(stgnf_dir)
    os.chdir(stgnf_dir)  # STG-NF's scorer reads data/ShanghaiTech/gt/... relative to its repo
    pose_root = os.path.join("data", "ShanghaiTech", "pose")

    log("Loading training poses...")
    train_xy, train_meta = load_split(gen_dataset, os.path.join(pose_root, "train"), SEG_LEN, args.train_stride)
    log("Loading test poses...")
    test_xy, test_meta = load_split(gen_dataset, os.path.join(pose_root, "test"), SEG_LEN, 1)

    # Hold out whole training clips (not random windows) for calibration, to avoid leakage
    clip_keys = np.array([f"{m[0]}_{m[1]}" for m in train_meta])
    clips = np.unique(clip_keys)
    val_clips = set(rng.choice(clips, size=max(1, int(len(clips) * args.val_frac)), replace=False))
    is_val = np.array([c in val_clips for c in clip_keys])
    log(f"Train windows {(~is_val).sum()} | calibration windows {is_val.sum()} "
        f"({len(val_clips)}/{len(clips)} clips) | test windows {len(test_xy)}")

    train_x = normalize_segments(train_xy[~is_val])
    val_x = normalize_segments(train_xy[is_val])
    test_x = normalize_segments(test_xy)

    arch_defaults = {
        "lstm": {"num_layers": 2, "lr": 1e-3},
        "transformer": {"num_layers": 3, "lr": 5e-4},
    }
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    pretty = {"lstm": "Drishti pose LSTM autoencoder",
              "transformer": "Drishti pose Transformer autoencoder"}

    results, curves, trained = {}, {}, {}

    def evaluate(name, normality):
        gt, scores = frame_level_eval(scoring_utils, normality, test_meta, SEG_LEN)
        auc = roc_auc_score(gt, scores)
        fpr, tpr, _ = roc_curve(1 - gt, -scores)
        curves[name] = (fpr, tpr, auc)
        results[name] = {"frame_auc": round(float(auc), 4)}
        log(f"{name}: frame-level AUC = {auc * 100:.2f}")
        return gt, scores

    gt, _ = evaluate("Speed heuristic (rule-based baseline)", -speed_baseline(test_xy))

    for arch in archs:
        cfg = dict(arch=arch, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, seg_len=SEG_LEN,
                   epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, **arch_defaults[arch])
        log(f"\n=== Training {pretty[arch]} ===")
        t0 = time.time()
        model = train_autoencoder(train_x, val_x, cfg, device, log)
        train_time = time.time() - t0

        val_err = batched_scores(model, val_x, args.batch_size)
        test_err = batched_scores(model, test_x, args.batch_size)
        threshold = float(np.percentile(val_err, args.threshold_pct))

        name = pretty[arch]
        gt, scores = evaluate(name, -test_err)
        anomaly_gt, anomaly_score = 1 - gt, -scores
        pred = (anomaly_score > threshold).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(anomaly_gt, pred, average="binary", zero_division=0)
        log(f"  at calibrated threshold: precision {pr:.3f} recall {rc:.3f} F1 {f1:.3f}")

        # CPU latency of the anomaly model alone for one person-window (what the app pays per person)
        cpu_model = build_model(cfg)
        cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
        cpu_model.eval()
        one = torch.from_numpy(test_x[:1])
        for _ in range(10):
            cpu_model.anomaly_score(one)
        t0 = time.time()
        for _ in range(200):
            cpu_model.anomaly_score(one)
        cpu_ms = (time.time() - t0) / 200 * 1000

        results[name].update({
            "threshold": threshold,
            "threshold_rule": f"{args.threshold_pct}th percentile of held-out normal training clips",
            "precision_at_threshold": round(float(pr), 4),
            "recall_at_threshold": round(float(rc), 4),
            "f1_at_threshold": round(float(f1), 4),
            "false_alarm_rate_on_normal_frames": round(float(pred[anomaly_gt == 0].mean()), 4),
            "params": int(sum(p.numel() for p in model.parameters())),
            "train_seconds": round(train_time, 1),
            "cpu_ms_per_window": round(cpu_ms, 2),
            "config": cfg,
        })
        trained[arch] = (cpu_model, cfg, threshold)

    summary = {
        "dataset": "ShanghaiTech Campus (frame-level, STG-NF pose release and scoring)",
        "reference_published": {"STG-NF (ICCV 2023), pose-only": 0.859},
        "methods": results,
        "deployed_model": pretty.get(args.deploy),
        "data": {"train_windows": int((~is_val).sum()), "calibration_windows": int(is_val.sum()),
                 "test_windows": int(len(test_xy)), "test_frames": int(len(gt)),
                 "anomalous_frame_ratio": round(float((1 - gt).mean()), 4)},
        "device": device,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for arch, (cpu_model, cfg, threshold) in trained.items():
        ckpt = {"state_dict": cpu_model.state_dict(),
                "config": {k: cfg[k] for k in ("arch", "hidden_dim", "latent_dim", "num_layers", "seg_len")},
                "threshold": threshold,
                "metrics": summary}
        torch.save(ckpt, os.path.join(args.out_dir, f"pose_ae_{arch}.pt"))
        if arch == args.deploy:
            torch.save(ckpt, os.path.join(args.out_dir, "pose_ae_shanghaitech.pt"))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC {auc * 100:.1f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ShanghaiTech frame-level ROC")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=150)

    log(json.dumps(summary, indent=2))
    log(f"Saved checkpoint and results to {args.out_dir}")


if __name__ == "__main__":
    main()

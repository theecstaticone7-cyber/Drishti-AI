import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F
import supervision as sv

# -----------------------------
# LOAD DETR MODEL
# -----------------------------
model = torch.hub.load(
    'facebookresearch/detr',
    'detr_resnet50',
    pretrained=True
)
model.eval()

# -----------------------------
# BYTE TRACKER
# -----------------------------
tracker = sv.ByteTrack()

# -----------------------------
# VIDEO SOURCE
# -----------------------------
USE_WEBCAM = True   # True = Webcam, False = Video file

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(
        "C:/Users/satwi/shoplifting-surveillance/inference/input_video.mp4"
    )

print("Video opened:", cap.isOpened())

# -----------------------------
# MEMORY FOR STABLE IDS
# -----------------------------
stable_ids = {}
next_stable_id = 0
previous_positions = {}

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to grab frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb).unsqueeze(0)

    # DETR inference
    with torch.no_grad():
        outputs = model(tensor)

    logits = outputs['pred_logits'][0]
    boxes = outputs['pred_boxes'][0]

    probas = logits.softmax(-1)
    scores, labels = probas.max(-1)

    h, w, _ = frame.shape

    detections = []

    # -----------------------------
    # DETECTION FILTERING
    # -----------------------------
    for score, label, box in zip(scores, labels, boxes):
        if score > 0.85 and label == 1:  # person

            cx, cy, bw, bh = box

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # Remove noise
            if (x2 - x1) < 40 or (y2 - y1) < 40:
                continue

            detections.append([x1, y1, x2, y2, score.item()])

    # -----------------------------
    # TRACKING + STABILIZATION
    # -----------------------------
    if len(detections) > 0:
        det = sv.Detections(
            xyxy=np.array([d[:4] for d in detections]),
            confidence=np.array([d[4] for d in detections])
        )

        tracks = tracker.update_with_detections(det)

        for xyxy, track_id in zip(tracks.xyxy, tracks.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)

            center = ((x1 + x2)//2, (y1 + y2)//2)

            # -----------------------------
            # STRONG ID STABILIZATION
            # -----------------------------
            assigned_id = None

            for sid, prev_center in stable_ids.items():
                dist = abs(center[0] - prev_center[0]) + abs(center[1] - prev_center[1])

                if dist < 80:   # key threshold
                    assigned_id = sid
                    break

            if assigned_id is None:
                assigned_id = next_stable_id
                next_stable_id += 1

            stable_ids[assigned_id] = center

            # -----------------------------
            # FILTER STATIC OBJECTS (DOLLS)
            # -----------------------------
            if assigned_id in previous_positions:
                prev = previous_positions[assigned_id]
                movement = abs(center[0] - prev[0]) + abs(center[1] - prev[1])

                if movement < 5:
                    continue

            previous_positions[assigned_id] = center

            # -----------------------------
            # DRAW OUTPUT
            # -----------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"ID {assigned_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    cv2.imshow("DETR + Stable Tracking (Final)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
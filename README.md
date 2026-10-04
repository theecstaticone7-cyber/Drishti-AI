# Drishti AI – AI-Powered Smart Surveillance System with Anomaly Detection

Drishti AI is an AI-powered smart surveillance system designed for real-time detection of suspicious activities and abnormal behavior in retail and public environments.

The system combines computer vision, deep learning, object detection, multi-object tracking, human pose estimation, and temporal behavior analysis to automatically identify potentially abnormal activities such as shoplifting, suspicious movements, object concealment, and unusual interactions.

Instead of relying entirely on continuous manual CCTV monitoring, Drishti AI analyzes video streams automatically, highlights suspicious events, generates alerts, and maintains an anomaly history for further analysis.

---

## 🚀 Key Features

- 🎥 Real-time video surveillance
- 📹 Recorded video and webcam input support
- 🎯 Transformer-based object detection using DETR
- 👥 Multi-object tracking using ByteTrack
- 🧍 Human pose estimation using MediaPipe
- 🔄 Temporal behaviour modelling with LSTM and Transformer autoencoders over 24-frame skeleton windows, fused with a motion-speed signal
- ⚠️ Unsupervised anomaly detection trained on normal motion only, benchmarked on ShanghaiTech Campus
- 🔔 Real-time visual anomaly alerts
- 📦 Bounding-box based anomaly visualization
- 🗃️ MongoDB-based anomaly storage
- 📸 Snapshot and incident history
- 📊 Interactive Streamlit dashboard
- 🔎 Anomaly history and monitoring

---

## 🧠 System Architecture

The overall processing pipeline is:

<img width="915" height="511" alt="image" src="https://github.com/user-attachments/assets/511f5669-f9c6-4769-91ef-06d1d6d7d342" />

Object Detection(DETR)
The object detection module is implemented using the DETR (Detection Transformer) model to identify people in video frames. 
Major Functions Used: 
• cv2.VideoCapture() – Captures video input
 • model.forward() – Performs object detection using 
• DETR post_process() – Filters outputs and extracts bounding boxes 
Working: Each video frame is passed to the DETR model, which outputs bounding boxes and confidence scores for detected objects. Only relevant classes (e.g., persons) are considered for further processing.
<img width="522" height="270" alt="image" src="https://github.com/user-attachments/assets/0bf8c52c-97e0-4c80-bb91-d8dfbea1502e" />

Tracking (ByteTrack)
The tracking module uses ByteTrack to maintain identity consistency of detected objects across frames. Major Functions Used: 
• tracker.update() – Updates object tracks 
• track_id assignment – Maintains unique ID for each person 
Working: Detected bounding boxes are passed to the tracker, which assigns unique IDs and tracks individuals across consecutive frames, even in crowded scenes.
<img width="574" height="350" alt="image" src="https://github.com/user-attachments/assets/b13d93d6-198b-4ca5-afb8-b382ad9b50dd" />

Pose Estimation (MediaPipe)
Each tracked person's box is padded by 10% and passed to MediaPipe Pose. The 33 MediaPipe landmarks are mapped to the 17 COCO joints (plus a neck point, 18 total) and converted back to full-frame pixel coordinates, so live poses use the same skeleton layout as the training data.
<img width="505" height="255" alt="image" src="https://github.com/user-attachments/assets/55e70b18-fbe1-4fec-babc-8bb1867d7add" />

Anomaly Detection Module (Pose Autoencoder + Speed Fusion)
Code: `inference/pose_anomaly.py`, training: `training/train_eval_shanghaitech.py`
• Each person keeps a sliding window of their last 24 skeletons.
• Each window is normalised (centred on the person and scaled by body height), so position and camera distance don't matter, only body shape and motion.
• An autoencoder trained **only on normal behaviour** tries to reconstruct the window. Motion it has never seen reconstructs badly, so reconstruction error flags unusual *body movement*. Two architectures are trained and compared: an LSTM autoencoder and a Transformer autoencoder (self-attention across the 24 frames, mean-pooled into a small latent bottleneck so it can't just copy its input).
• A rule-based speed score (joint speed divided by body size) flags *fast* movement such as running or cycling.
• The final score fuses both: each is standardised using held-out normal training clips and they are added with equal weights. Nothing is tuned on the test set.
• The alert threshold is the 99th percentile of errors on held-out normal training clips, so it is calibrated on data, not hand-tuned.
• An alert fires only after several consecutive anomalous windows (hysteresis), which cuts one-frame false alarms. The person, time and snapshot are logged to MongoDB.

Alert and Visualization(Streamlit)
This module handles real-time display and alert generation. 
Major Functions Used: 
• st.image() / st.video() – Displays video output 
• st.warning() / st.alert() – Shows anomaly alerts 
• json.dump() – Stores results 
Working: The processed video is displayed using Streamlit, highlighting anomalies with bounding boxes. Alerts are generated in real time, and results are stored for reporting. 
<img width="638" height="309" alt="image" src="https://github.com/user-attachments/assets/e8f2b791-6dd5-4803-9065-cc993cf5ab54" />

---

## 📈 Evaluation on ShanghaiTech Campus

The anomaly model is trained and evaluated with the standard ShanghaiTech Campus protocol: train on normal clips only, score every test frame, and report frame-level ROC-AUC. It uses the pose data, ground truth and scoring code released with STG-NF (ICCV 2023), so the numbers are directly comparable to the published result.

| Method | Frame-level AUC | Precision* | Recall* | False alarms* |
|---|---|---|---|---|
| Speed heuristic (rule-based baseline) | 78.5 | 0.82 | 0.13 | 2.2% |
| LSTM autoencoder | 77.0 | 0.76 | 0.10 | 2.4% |
| Transformer autoencoder | 74.2 | 0.72 | 0.08 | 2.3% |
| **LSTM autoencoder + speed (fusion, deployed)** | **79.1** | 0.78 | 0.11 | 2.2% |
| Transformer autoencoder + speed (fusion) | 77.8 | 0.71 | 0.08 | 2.3% |
| STG-NF (ICCV 2023, published, pose-only) | 85.9 | – | – | – |

\* At the alert threshold (99th percentile of scores on held-out normal training clips). Test set: 40,791 frames, 42% anomalous.

**What the numbers say:**
- On its own, neither learned model beats the simple speed rule. Most ShanghaiTech anomalies are people cycling, running or skateboarding, so speed alone is a strong signal.
- Fusing the LSTM autoencoder with speed gives the best AUC (79.1, +0.6 over speed alone), so the autoencoder adds a small amount of information that speed misses. This is from one training run, so treat the gain as modest.
- The LSTM beat the Transformer on every metric while being 2.4× smaller and about 2× faster on CPU (1.9 vs 4.5 ms per person-window). With 24-frame windows and roughly 130k training windows, self-attention brought no advantage here.
- At the strict alert threshold, the system catches about 11% of anomalous frames with about 2% false alarms on normal frames. It's tuned to avoid crying wolf, and a lower threshold trades more catches for more false alarms.
- STG-NF's normalizing-flow model is still about 7 AUC points ahead.

Precision, recall, F1 and false-alarm rate at the calibrated alert threshold are in `inference/models/results.json`, and the ROC curve is in `inference/models/roc_curve.png`.

### Reproduce
Open `notebooks/train_eval_shanghaitech.ipynb` in Google Colab with a GPU runtime and run all cells (about 30 minutes). It downloads the data, trains the model, prints the table above, and downloads the checkpoints. Put them in `inference/models/`; the app loads `pose_ae_shanghaitech.pt` (LSTM + speed fusion by default).

## ▶️ Run the app
```bash
cd inference
pip install -r requirements.txt
# MongoDB must be running on localhost:27017 for incident history
streamlit run app.py
```

## ⚠️ Limitations
- The model is trained on AlphaPose skeletons and runs on MediaPipe skeletons at inference time. Both use the same COCO joint layout, but the keypoint detectors differ, so live accuracy may be lower than the benchmark.
- Anomaly here means *unusual body motion* compared with ShanghaiTech's campus scenes. It does not understand objects or intent, so concealment of an item without unusual movement is not detected.
- DETR on CPU is slow, so a GPU is needed for real-time frame rates.
- It's a decision-support tool for a human operator, not an automatic accusation system.

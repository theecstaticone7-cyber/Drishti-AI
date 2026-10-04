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
- 🧠 Spatial feature extraction using ResNet18
- 🔄 Temporal behavior analysis using Transformer Encoder
- ⚠️ Hybrid anomaly detection using model predictions and motion analysis
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

Pose Estimation(MediaPipe)
This module performs human pose analysis using MediaPipe for understanding posture and gesture patterns. In this system, pose estimation works implicitly without displaying skeletal keypoints or pose lines on the output. 
Major Functions Used: 
• mp.solutions.pose.Pose() – Initializes pose model 
• pose.process() – Processes frames to extract pose-related features 
Working: The module internally analyzes human body movements by extracting keypoint information such as joints and limb positions. These features help in identifying actions like bending, unusual hand movements, or object concealment. Instead of visualizing pose landmarks, the extracted information is directly used to improve behavior analysis in the anomaly detection process. 
<img width="505" height="255" alt="image" src="https://github.com/user-attachments/assets/55e70b18-fbe1-4fec-babc-8bb1867d7add" />

Anomaly Detection Module (ResNet + Transformer) 
This module identifies abnormal behavior using spatial and temporal analysis. 
Major Functions Used: 
• resnet_model() – Extracts spatial features 
• transformer_encoder() – Processes temporal sequences 
• calculate_motion_diff() – Computes motion differences 
• threshold_check() – Classifies anomaly 
Working: A sequence of frames is processed to extract features using ResNet18. These features are passed through a Transformer Encoder to analyze motion patterns. A hybrid approach combining transformer output and motion difference is used to detect anomalies. 

<img width="485" height="277" alt="image" src="https://github.com/user-attachments/assets/4b482607-2d10-4c4e-94d5-07d0002bffe5" />

Alert and Visualization(Streamlit)
This module handles real-time display and alert generation. 
Major Functions Used: 
• st.image() / st.video() – Displays video output 
• st.warning() / st.alert() – Shows anomaly alerts 
• json.dump() – Stores results 
Working: The processed video is displayed using Streamlit, highlighting anomalies with bounding boxes. Alerts are generated in real time, and results are stored for reporting. 
<img width="638" height="309" alt="image" src="https://github.com/user-attachments/assets/e8f2b791-6dd5-4803-9065-cc993cf5ab54" />














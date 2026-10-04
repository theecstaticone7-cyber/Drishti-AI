import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from torchvision.transforms import functional as F
import supervision as sv
import mediapipe as mp
import json
from datetime import datetime
import os
import pandas as pd
import io

# DB
from mongo_db import store_anomaly, get_all_anomalies

# UI
from ui import setup_ui

confidence_threshold, video_source, start, frame_placeholder = setup_ui()
alert_box = st.empty()

# SNAPSHOT FOLDER
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# SIDEBAR BUTTON
with st.sidebar:
    show_history = st.button("📜 VIEW ANOMALY HISTORY", use_container_width=True)

@st.cache_resource
def load_model():
    model = torch.hub.load('facebookresearch/detr','detr_resnet50',pretrained=True)
    model.eval()
    return model

model = load_model()

class VideoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256,nhead=4,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)
        features = self.backbone(x)
        features = features.view(B,T,512)
        features = self.proj(features)
        x = self.transformer(features)
        x = x.mean(dim=1)
        return self.fc(x)

@st.cache_resource
def load_anomaly_model():
    model = VideoTransformer()
    model.eval()
    return model

anomaly_model = load_anomaly_model()

tracker = sv.ByteTrack()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

frame_sequences = {}
alert_counter = {}

stored_ids = {}

frame_skip = 2
frame_count = 0

def save_incident(track_id):
    data = {
        "track_id": int(track_id),
        "time": str(datetime.now()),
        "event": "Suspicious Activity"
    }
    with open("incidents.json", "a") as f:
        f.write(json.dumps(data) + "\n")

def generate_alert(track_id):
    return f"Person {track_id} showing suspicious behavior!"

if start and video_source is not None:

    cap = cv2.VideoCapture(video_source)

    while cap.isOpened() and st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)

        logits = outputs['pred_logits'][0]
        boxes = outputs['pred_boxes'][0]

        probas = logits.softmax(-1)
        scores, labels = probas.max(-1)

        h, w, _ = frame.shape
        detections = []

        for score, label, box in zip(scores, labels, boxes):
            if score > confidence_threshold and label == 1:

                cx, cy, bw, bh = box
                x1 = int((cx - bw/2)*w)
                y1 = int((cy - bh/2)*h)
                x2 = int((cx + bw/2)*w)
                y2 = int((cy + bh/2)*h)

                x1,y1 = max(0,x1),max(0,y1)
                x2,y2 = min(w,x2),min(h,y2)

                if (x2-x1)>50 and (y2-y1)>50:
                    detections.append([x1,y1,x2,y2,score.item()])

        if len(detections)>0:

            det = sv.Detections(
                xyxy=np.array([d[:4] for d in detections]),
                confidence=np.array([d[4] for d in detections])
            )

            tracks = tracker.update_with_detections(det)

            for xyxy, track_id in zip(tracks.xyxy, tracks.tracker_id):

                x1,y1,x2,y2 = map(int,xyxy)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"ID {track_id}",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

                crop = rgb[y1:y2,x1:x2]
                if crop.size==0:
                    continue

                crop = cv2.resize(crop,(112,112))
                crop_tensor = torch.tensor(crop).permute(2,0,1).float()/255.0

                if track_id not in frame_sequences:
                    frame_sequences[track_id]=[]
                    alert_counter[track_id]=0

                frame_sequences[track_id].append(crop_tensor)

                if len(frame_sequences[track_id])==16:

                    seq = torch.stack(frame_sequences[track_id]).unsqueeze(0)

                    with torch.no_grad():
                        output = anomaly_model(seq)

                    prob = torch.softmax(output,dim=1)
                    transformer_score = prob[0][1].item()

                    seq_np = seq.squeeze(0).numpy()
                    diffs = np.mean(np.abs(seq_np[1:] - seq_np[:-1]))

                    if transformer_score>0.5 and diffs>0.08:
                        alert_counter[track_id]+=1
                    else:
                        alert_counter[track_id]=max(0,alert_counter[track_id]-1)

                    if alert_counter[track_id] > 3:

                        alert_msg = generate_alert(track_id)

                        cv2.putText(frame,"🚨 Suspicious Activity",(x1,y2+20),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                        alert_box.error(f"🚨 ALERT: {alert_msg}")

                        save_incident(track_id)

                        if track_id not in stored_ids:

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_path = f"{SNAPSHOT_DIR}/person_{track_id}_{timestamp}.jpg"
                            cv2.imwrite(image_path, frame)

                            store_anomaly(video_source, track_id, image_path)

                            stored_ids[track_id] = True

                    frame_sequences[track_id].pop(0)

        frame_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            width=700
        )

    cap.release()

# ================= HISTORY SECTION =================

if show_history:

    st.markdown("## 📜 Anomaly History")

    data = get_all_anomalies()

    if len(data) == 0:
        st.info("No anomalies recorded yet.")
    else:
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # FILTER
        st.markdown("### 📅 Filter by Date")

        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()

        start_date, end_date = st.date_input(
            "Select Date Range",
            [min_date, max_date]
        )

        filtered_df = df[
            (df["timestamp"].dt.date >= start_date) &
            (df["timestamp"].dt.date <= end_date)
        ]

        # ================= REPORT (FIXED) =================
        st.markdown("### 📄 Generate Report")

        report_type = st.radio(
            "Select Report Type",
            ["Filtered Data", "Full Data"],
            horizontal=True
        )

        report_df = filtered_df if report_type == "Filtered Data" else df

        if len(report_df) == 0:
            st.warning("No data available for report.")
        else:
            csv_buffer = io.StringIO()
            report_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="⬇️ Download Report",
                data=csv_buffer.getvalue(),
                file_name="anomaly_report.csv",
                mime="text/csv"
            )

        # CHART
        st.markdown("### 📊 Anomaly Trend")

        chart_data = filtered_df.copy()
        chart_data["date"] = chart_data["timestamp"].dt.date
        trend = chart_data.groupby("date").size()

        st.line_chart(trend)

        # DISPLAY
        st.markdown("### 📂 Records")

        for _, item in filtered_df.iterrows():

            st.markdown("---")

            col1, col2 = st.columns([1,2])

            with col1:
                if item.get("snapshot") and os.path.exists(item["snapshot"]):
                    st.image(item["snapshot"], caption="Snapshot", use_container_width=True)
                else:
                    st.warning("No Image")

            with col2:
                st.write(f"👤 **Person ID:** {item.get('person_id')}")
                st.write(f"⏱ **Time:** {item.get('timestamp')}")
                st.write(f"🎥 **Video:** {item.get('video_path')}")
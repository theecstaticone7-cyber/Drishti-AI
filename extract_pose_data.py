import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            sequence.append(keypoints)

    cap.release()
    return sequence


def process_dataset(folder, label):
    data = []
    labels = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        seq = extract_from_video(path)

        if len(seq) > 10:
            data.append(seq[:30])
            labels.append(label)

    return data, labels


normal_data, normal_labels = process_dataset("dataset/normal", 0)
abnormal_data, abnormal_labels = process_dataset("dataset/abnormal", 1)

X = normal_data + abnormal_data
y = normal_labels + abnormal_labels

np.save("X.npy", np.array(X, dtype=object))
np.save("y.npy", np.array(y))

print("✅ Data extraction complete")
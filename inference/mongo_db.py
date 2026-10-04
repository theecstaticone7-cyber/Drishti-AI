from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["surveillance_db"]
collection = db["anomalies"]

# ✅ STORE FUNCTION
def store_anomaly(video_path, track_id, image_path=None):
    data = {
        "anomaly": True,
        "timestamp": datetime.now(),
        "video_path": video_path,
        "person_id": int(track_id),
        "snapshot": image_path
    }

    collection.insert_one(data)
    print("✅ Stored in MongoDB:", data)

# ✅ NEW FUNCTION (FETCH HISTORY)
def get_all_anomalies():
    return list(collection.find().sort("timestamp", -1))
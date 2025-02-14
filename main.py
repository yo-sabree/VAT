import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import time

model = YOLO('yolov8l.pt')
tracker = DeepSort(max_age=30)
cred = credentials.Certificate("E:/BRiX/vat-demo-f206b-firebase-adminsdk-fbsvc-e8d3543a25.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
video_path = "E:/BRiX/Brix/VAT/Samples/Sample1.mp4"
cap = cv2.VideoCapture(video_path)
interaction_data = defaultdict(
    lambda: {"interactions": defaultdict(int), "total_time": 0, "first_detected": None, "last_detected": None,
             "contacted_objects": set(), "person_interactions": set()})
last_update_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (800, 600))
    results = model(frame)
    detections = []
    objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == 'person':
                detections.append(([x1, y1, x2, y2], box.conf[0].item(), None))
            else:
                objects.append((box.xyxy[0], label))

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltwh())
        if track_id not in interaction_data:
            interaction_data[track_id]["first_detected"] = datetime.now()
        interaction_data[track_id]["last_detected"] = datetime.now()
        for obj_bbox, obj_label in objects:
            ox1, oy1, ox2, oy2 = map(int, obj_bbox)
            if (x1 < ox2 and x2 > ox1) and (y1 < oy2 and y2 > oy1):
                if obj_label not in interaction_data[track_id]["contacted_objects"]:
                    interaction_data[track_id]["interactions"][obj_label] += 1
                    interaction_data[track_id]["contacted_objects"].add(obj_label)
        for other_track in tracks:
            other_id = other_track.track_id
            if track_id != other_id:
                ox1, oy1, ox2, oy2 = map(int, other_track.to_ltwh())
                if (x1 < ox2 and x2 > ox1) and (y1 < oy2 and y2 > oy1):
                    interaction_data[track_id]["person_interactions"].add(f"person_{other_id}")

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltwh())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for obj_bbox, obj_label in objects:
        x1, y1, x2, y2 = map(int, obj_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, obj_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("YOLO DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if time.time() - last_update_time >= 5:
        for pid, data in interaction_data.items():
            total_time = (data["last_detected"] - data["first_detected"]).total_seconds() if data["first_detected"] and \
                                                                                             data[
                                                                                                 "last_detected"] else 0
            doc_ref = db.collection("interaction_data").document(f"person_{pid}")
            doc_ref.set({
                "person_id": pid,
                "first_detected": data["first_detected"],
                "last_detected": data["last_detected"],
                "total_time": total_time,
                "interactions": dict(data["interactions"]),
                "person_interactions": list(data["person_interactions"]),
                "timestamp": datetime.now()
            })
        last_update_time = time.time()

cap.release()
cv2.destroyAllWindows()
print("Interaction data has been stored in Firestore.")

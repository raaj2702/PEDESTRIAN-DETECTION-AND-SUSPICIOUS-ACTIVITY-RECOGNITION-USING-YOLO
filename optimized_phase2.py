
import cv2
import time
import numpy as np
import json
import torch
from ultralytics import YOLO

#CONFIGURATION 
VIDEO_PATH = "bus.mp4"
OUTPUT_PATH = "output_activity.mp4"
LOG_PATH = "suspicious_log.json"
CONFIDENCE_THRESHOLD = 0.4
LOITER_FRAMES = 50
RUNNING_DISTANCE = 80
FALLING_ANGLE_THRESHOLD = 40  # degrees
CROWD_THRESHOLD = 4  # people within close proximity
SHOW_RESTRICTED_ZONE = True
ZONE_TOP_LEFT = (400, 100)
ZONE_BOTTOM_RIGHT = (600, 300)
ZONE_COLOR = (255, 0, 255)
ZONE_LABEL = "Restricted Area"

# ------------------------ INITIALIZATION ------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    yolo_model = YOLO("yolov8n.pt").to(device)
except Exception as e:
    print(f"Model load failed: {e}")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Failed to open video file: {VIDEO_PATH}")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

person_id_counter = 0
person_tracks = {}
suspicious_log = []
frame_number = 0

# ------------------------ HELPER FUNCTIONS ------------------------
def is_inside_zone(center):
    return (ZONE_TOP_LEFT[0] <= center[0] <= ZONE_BOTTOM_RIGHT[0] and
            ZONE_TOP_LEFT[1] <= center[1] <= ZONE_BOTTOM_RIGHT[1])

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return np.degrees(np.arctan2(height, width))

# ------------------------ MAIN PROCESSING LOOP ------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]
    people = [box for box in results.boxes if int(box.cls[0]) == 0 and float(box.conf[0]) > CONFIDENCE_THRESHOLD]
    centers = []

    for box in people:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        centers.append(center)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{float(box.conf[0]):.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Track loitering and running
        matched = False
        for pid, track in person_tracks.items():
            if distance(center, track[-1]) < 50:
                track.append(center)
                if len(track) > LOITER_FRAMES and distance(track[-1], track[-LOITER_FRAMES]) < 20:
                    cv2.putText(frame, "LOITERING", (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    suspicious_log.append({"frame": frame_number, "event": "loitering", "x": center[0], "y": center[1]})
                if len(track) > 5 and distance(track[-1], track[-5]) > RUNNING_DISTANCE:
                    cv2.putText(frame, "RUNNING", (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    suspicious_log.append({"frame": frame_number, "event": "running", "x": center[0], "y": center[1]})
                matched = True
                break
        if not matched:
            person_tracks[person_id_counter] = [center]
            person_id_counter += 1

        # Fall detection
        angle = calculate_angle((x1, y1, x2, y2))
        if angle < FALLING_ANGLE_THRESHOLD:
            cv2.putText(frame, "FALLING", (x1, y2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            suspicious_log.append({"frame": frame_number, "event": "falling", "x": center[0], "y": center[1]})

        # Zone intrusion
        if SHOW_RESTRICTED_ZONE and is_inside_zone(center):
            cv2.putText(frame, "ZONE INTRUSION", (x1, y2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_COLOR, 1)
            suspicious_log.append({"frame": frame_number, "event": "zone_intrusion", "x": center[0], "y": center[1]})

    # Crowding detection
    for i in range(len(centers)):
        close = 1
        for j in range(len(centers)):
            if i != j and distance(centers[i], centers[j]) < 60:
                close += 1
        if close >= CROWD_THRESHOLD:
            cv2.putText(frame, "CROWDING", centers[i],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            suspicious_log.append({"frame": frame_number, "event": "crowding", "x": centers[i][0], "y": centers[i][1]})
            break

    # Draw zone box
    if SHOW_RESTRICTED_ZONE:
        cv2.rectangle(frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, ZONE_COLOR, 2)
        cv2.putText(frame, ZONE_LABEL, (ZONE_TOP_LEFT[0], ZONE_TOP_LEFT[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_COLOR, 1)

    out.write(frame)
    cv2.imshow("Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

# ------------------------ SAVE & CLEANUP ------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

try:
    with open(LOG_PATH, 'w') as f:
        json.dump(suspicious_log, f, indent=4)
    print(f"Log saved to {LOG_PATH}")
except Exception as e:
    print(f"Error saving log: {e}")

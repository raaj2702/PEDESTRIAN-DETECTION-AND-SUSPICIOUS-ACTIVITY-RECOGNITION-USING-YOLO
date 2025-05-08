import cv2
import time
import numpy as np
import json
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Detection settings
CONFIDENCE_THRESHOLD = 0.4
LOITER_FRAMES = 10
RUNNING_DISTANCE = 20
FALLING_ANGLE_THRESHOLD = 40  # degrees
CROWD_THRESHOLD = 4  # number of people in close proximity

# Tracking variables
person_id_counter = 0
person_tracks = {}
suspicious_log = []
frame_number = 0

# Optional zone setup
ZONE_TOP_LEFT = (400, 100)
ZONE_BOTTOM_RIGHT = (600, 300)

# Video input/output
cap = cv2.VideoCapture("susvid.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_activity.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

def is_inside_zone(center):
    return (ZONE_TOP_LEFT[0] <= center[0] <= ZONE_BOTTOM_RIGHT[0] and
            ZONE_TOP_LEFT[1] <= center[1] <= ZONE_BOTTOM_RIGHT[1])

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if height == 0:
        return 90
    angle = np.degrees(np.arctan(width / height))
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    detections = yolo_model(frame)[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    scores = detections.boxes.conf.cpu().numpy()
    classes = detections.boxes.cls.cpu().numpy()

    current_centers = []
    current_boxes = []

    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        if score < CONFIDENCE_THRESHOLD or int(cls) != 0:
            continue
        x1, y1, x2, y2 = map(int, box)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        current_centers.append(center)
        current_boxes.append(box)

        matched = False
        for pid, data in person_tracks.items():
            prev_center = data['center']
            if distance(center, prev_center) < 50:
                data['center'] = center
                data['frames'] += 1
                matched = True

                if data['frames'] > LOITER_FRAMES:
                    cv2.putText(frame, "LOITERING DETECTED", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    suspicious_log.append({"frame": frame_number, "type": "loitering", "location": center})

                if distance(center, prev_center) > RUNNING_DISTANCE:
                    cv2.putText(frame, "RUNNING DETECTED", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    suspicious_log.append({"frame": frame_number, "type": "running", "location": center})

                if is_inside_zone(center):
                    cv2.putText(frame, "ZONE VIOLATION", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    suspicious_log.append({"frame": frame_number, "type": "zone_violation", "location": center})
                break

        if not matched:
            person_id_counter += 1
            person_tracks[person_id_counter] = {"center": center, "frames": 1}

        angle = calculate_angle([x1, y1, x2, y2])
        if angle > FALLING_ANGLE_THRESHOLD:
            cv2.putText(frame, "FALLING DETECTED", (x1, y1 - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
            suspicious_log.append({"frame": frame_number, "type": "falling", "location": center})

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Crowd detection
    for i in range(len(current_centers)):
        crowd_count = 1
        for j in range(i + 1, len(current_centers)):
            if distance(current_centers[i], current_centers[j]) < 70:
                crowd_count += 1
        if crowd_count >= CROWD_THRESHOLD:
            cv2.putText(frame, "CROWDING DETECTED", (current_centers[i][0], current_centers[i][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            suspicious_log.append({"frame": frame_number, "type": "crowding", "location": current_centers[i]})

    # Draw zone
    cv2.rectangle(frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, (255, 0, 0), 2)
    cv2.putText(frame, f"Frame: {frame_number} | People: {len(current_centers)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Write and display
    out.write(frame)
    cv2.imshow("YOLOv8 Suspicious Activity Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save suspicious log
with open("suspicious_log.json", "w") as f:
    json.dump(suspicious_log, f, indent=4)

print("\nDetection complete. Log saved as 'suspicious_log.json'")

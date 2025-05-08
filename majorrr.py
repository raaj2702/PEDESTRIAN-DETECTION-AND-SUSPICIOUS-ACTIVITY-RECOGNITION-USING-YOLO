import torch
import torchvision
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Settings
CONFIDENCE_THRESHOLD = 0.5
DEVICE = torch.device('cpu')
COCO_LABELS = ['person', 'car', 'motorcycle', 'bus', 'truck']

# Load models
yolo_model = YOLO('yolov8n.pt')
ssd_model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT").to(DEVICE)
ssd_model.eval()
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE)
faster_rcnn_model.eval()

# Track FPS and detections
fps_tracker = {"YOLOv8": [], "SSD": [], "Faster R-CNN": []}
total_detections = {"YOLOv8": 0, "SSD": 0, "Faster R-CNN": 0}

def draw_boxes(frame, boxes, labels, scores, fps, model_name):
    for box, label, score in zip(boxes, labels, scores):
        if score > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{label} {score:.2f}'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'{model_name} - FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return frame

def detect_with_model(frame, model_name):
    labels, boxes, scores = [], [], []
    start = time.time()

    if model_name == "YOLOv8":
        results = yolo_model(frame)
        raw_scores = results[0].boxes.conf.cpu().numpy()
        raw_boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        for box, cls, score in zip(raw_boxes, classes, raw_scores):
            if score > CONFIDENCE_THRESHOLD and int(cls) < len(COCO_LABELS):
                label = COCO_LABELS[int(cls)]
                if label == "person":
                    boxes.append(box)
                    labels.append(label)
                    scores.append(score)

    elif model_name == "SSD":
        tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            detections = ssd_model(tensor)[0]
        for box, cls, score in zip(detections['boxes'], detections['labels'], detections['scores']):
            if score > CONFIDENCE_THRESHOLD and int(cls) < len(COCO_LABELS):
                label = COCO_LABELS[int(cls)]
                if label == "person":
                    boxes.append(box.cpu().numpy())
                    labels.append(label)
                    scores.append(score.cpu().numpy())

    elif model_name == "Faster R-CNN":
        tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            detections = faster_rcnn_model(tensor)[0]
        for box, cls, score in zip(detections['boxes'], detections['labels'], detections['scores']):
            if score > CONFIDENCE_THRESHOLD and int(cls) < len(COCO_LABELS):
                label = COCO_LABELS[int(cls)]
                if label == "person":
                    boxes.append(box.cpu().numpy())
                    labels.append(label)
                    scores.append(score.cpu().numpy())

    fps = 1 / (time.time() - start)
    fps_tracker[model_name].append(fps)
    total_detections[model_name] += len(labels)
    return draw_boxes(frame.copy(), boxes, labels, scores, fps, model_name)

def real_time_detection(video_source='your_new_video.mp4'):
    cap = cv2.VideoCapture(video_source)
    current_model = "YOLOv8"
    print("Press 1 = YOLOv8 | 2 = SSD | 3 = Faster R-CNN | q = Quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = detect_with_model(frame, current_model)
        cv2.imshow("Pedestrian Detection", result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_model = "YOLOv8"
        elif key == ord('2'):
            current_model = "SSD"
        elif key == ord('3'):
            current_model = "Faster R-CNN"
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot FPS comparison
    plt.figure(figsize=(10, 6))
    for model, data in fps_tracker.items():
        if data:
            plt.plot(data, label=model)
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.title('FPS Comparison of Models')
    plt.legend()
    plt.grid(True)
    plt.savefig("fps_comparison.png")
    plt.show()

    # Print performance summary
    print("\nModel Performance Summary:")
    for model in fps_tracker:
        avg_fps = np.mean(fps_tracker[model]) if fps_tracker[model] else 0
        total = total_detections[model]
        print(f"{model:<15} | Avg FPS: {avg_fps:.2f} | Total People Detected: {total}")

real_time_detection(video_source='pedvid.mp4')
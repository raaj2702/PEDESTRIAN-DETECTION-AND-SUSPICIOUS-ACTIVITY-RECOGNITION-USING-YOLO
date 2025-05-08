import cv2
import torch
import torchvision
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import numpy as np
import json

device = torch.device('cpu')

yolo_model = YOLO('yolov8n.pt')

ssd_model = torchvision.models.detection.ssd300_vgg16(pretrained=True).to(device)
ssd_model.eval()

faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
faster_rcnn_model.eval()

coco_labels = [
    'person',  'car', 'motorcycle',  'bus', 'truck'
]

coco_gt = COCO('instances_val.json')



def detect_yolo(frame):
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    return boxes, classes



def detect_ssd(frame):
    frame_tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        detections = ssd_model(frame_tensor)[0]
    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    return boxes, labels



def detect_faster_rcnn(frame):
    frame_tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        detections = faster_rcnn_model(frame_tensor)[0]
    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    return boxes, labels



def draw_boxes(frame, boxes, labels, scores, fps):
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.8 :
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{label} {score:.2f}'  
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame



def evaluate_model(coco_gt, results_path):
    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



def real_time_detection(video_source=0, output_file='output.avi'):
    cap = cv2.VideoCapture(video_source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    yolo_results = []
    ssd_results = []
    frcnn_results = []

    category_map = {
        'person': 0,
        'car': 1,
        'motorcycle': 2,
        'bus': 3,
        'truck': 4
    }

    coco_image_ids = coco_gt.getImgIds()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img_id = coco_image_ids[frame_count % len(coco_image_ids)]

        start_time = time.time()

        
        results = yolo_model(frame)
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy() 
        yolo_scores = results[0].boxes.conf.cpu().numpy()
        yolo_classes = results[0].boxes.cls.cpu().numpy() 
        yolo_labels = [coco_labels[int(cls)] for cls in yolo_classes if int(cls) < len(coco_labels)]

       
        frame_tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            detections = ssd_model(frame_tensor)[0]
        ssd_boxes = detections['boxes'].cpu().numpy() 
        ssd_scores = detections['scores'].cpu().numpy() 
        ssd_classes = detections['labels'].cpu().numpy() 

        ssd_labels = []
        for cls in ssd_classes:
            if str(cls).isdigit():  
                cls_int = int(cls)
                if cls_int == 1:
                    cls_int= 0
                if cls_int < len(coco_labels): 
                    ssd_labels.append(coco_labels[cls_int])

       
        frame_tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            detections = faster_rcnn_model(frame_tensor)[0]
        frcnn_boxes = detections['boxes'].cpu().numpy()  
        frcnn_scores = detections['scores'].cpu().numpy()  
        frcnn_classes = detections['labels'].cpu().numpy()  
        frcnn_labels = []
        for cls in frcnn_classes:
            if str(cls).isdigit():  
                cls_int = int(cls)
                if cls_int == 1:
                    cls_int= 0
                if cls_int < len(coco_labels):  
                    frcnn_labels.append(coco_labels[cls_int])

        end_time = time.time()
        fps = 1 / (end_time - start_time)

        
        frame_yolo = draw_boxes(frame.copy(), yolo_boxes, yolo_labels, yolo_scores, fps)
        frame_ssd = draw_boxes(frame.copy(), ssd_boxes, ssd_labels, ssd_scores, fps)
        frame_frcnn = draw_boxes(frame.copy(), frcnn_boxes, frcnn_labels, frcnn_scores, fps)

        
        out.write(frame_yolo)

        #
        
        cv2.imshow('YOLOv8 Detection', frame_yolo)
        cv2.imshow('SSD Detection', frame_ssd)
        cv2.imshow('Faster R-CNN Detection', frame_frcnn)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


real_time_detection(video_source='bus.jpg', output_file='output.avi')
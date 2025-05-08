# PEDESTRIAN-DETECTION-AND-SUSPICIOUS-ACTIVITY-RECOGNITION-USING-YOLO

This project utilizes YOLOv8 for detecting pedestrians and identifying suspicious activities in surveillance videos. It supports real-time video input, performs detection using pretrained models (YOLOv8, SSD, Faster R-CNN), and logs suspicious behavior for analysis.

## 📂 Project Structure
- `majorrr.py` (main script for pedestrian detection and activity recognition)
- `yolov8n.pt` (YOLOv8 nano pre-trained weights)
- `bus.mp4`, `susvid.mp4`, `pedvid.mp4` (sample input videos)
- `output.avi`, `output_activity.mp4` (detection output videos)
- `suspicious_log.json` (logged suspicious activity data)
- `fps_comparison.png` (performance comparison image)

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

import cv2
from ultralytics import YOLO
import os

# Paths
video_path = os.path.join('.', 'deppheard.mp4')
video_out_path = os.path.join('.', 'out.mp4')

# Video capture
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Video writer
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# Load YOLO model
model = YOLO("yolov8n.pt")

# Detection threshold
detection_threshold = 0.5
constant_color = (0, 255, 0)  # Green color for rectangles

while ret:
    # Get model results
    results = model(frame, verbose=False)
    
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold and class_id == 0:
                detections.append([x1, y1, x2, y2, score])
                # Draw rectangle for people detections
                cv2.rectangle(frame, (x1, y1), (x2, y2), constant_color, 3)

    # Write frame with detections
    cap_out.write(frame)
    ret, frame = cap.read()

# Release resources
cap.release()
cap_out.release()
cv2.destroyAllWindows()

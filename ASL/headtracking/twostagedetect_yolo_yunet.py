from yunetdetecttest import FaceDetectorYunet
import cv2
from ultralytics import YOLO
import os
from tqdm import tqdm


# Initialize YOLO and FaceDetectorYunet
yolo_model = YOLO("yolov8n.pt")
fd = FaceDetectorYunet()

input_video_path = 'deppheard.mp4'
output_video_path = '2stage_yolo_yunet_detect_test.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frames, desc="Processing frames")


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Detection threshold
detection_threshold = 0.5
yolo_color = (0, 0, 255)  # Red color for YOLO detections

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO person detection
    results = yolo_model(frame, verbose = False)
    person_detections = []

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > detection_threshold and int(class_id) == 0:  # Class ID 0 is for persons in COCO dataset
                person_detections.append((int(x1), int(y1), int(x2), int(y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), yolo_color, 2)

    # Process each detected person
    for (x1, y1, x2, y2) in person_detections:
        person_roi = frame[y1:y2, x1:x2]  # Extract region of interest
        faces = fd.detect(person_roi)  # Detect faces within the person ROI

        if faces:
            for face in faces:
                # Adjust face coordinates relative to the original frame
                face['x1'] += x1
                face['y1'] += y1
                face['x2'] += x1
                face['y2'] += y1
            # Draw face detections on the original frame
            fd.draw_faces(frame, faces)

    # Write the frame with face detections
    out.write(frame)
    pbar.update(1)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
pbar.close()
import cv2
import face_recognition
from ultralytics import YOLO
import os
from tqdm import tqdm
import numpy as np

# Function to encode a face from an image
def encode_face(image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    return encoding

# Function to perform face recognition on a video
def recognize_faces_in_video(input_video_path, output_video_path, known_face_encodings, known_face_names):
    # Video capture
    video_capture = cv2.VideoCapture(input_video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing frames")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")

    detection_threshold = 0.5
    constant_color = (0, 255, 0)  # Green color for face rectangles

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Get YOLO detections
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for YOLO person

                    # Perform face recognition on the detected region
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cropped_rgb_frame = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])
                    face_locations = face_recognition.face_locations(cropped_rgb_frame)
                    face_encodings = face_recognition.face_encodings(cropped_rgb_frame, face_locations)

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = face_distances.argmin()
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        cv2.rectangle(frame, (left+x1, top+y1), (right+x1, bottom+y1), (0, 255, 0), 2)  # Green box for face
                        cv2.putText(frame, name, (left+x1 + 6, bottom+y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        output_video.write(frame)
        pbar.update(1)

    video_capture.release()
    output_video.release()
    pbar.close()

# Encode faces of known people
person1_image_path = '../face_assoc/audio_face/Johnny_Depp.jpg'
person2_image_path = '../face_assoc/audio_face/jd2.png'
person3_image_path = '../face_assoc/audio_face/jd3.png'
person4_image_path = '../face_assoc/audio_face/ah.png'
person1_encoding = encode_face(person1_image_path)
person2_encoding = encode_face(person2_image_path)
person3_encoding = encode_face(person3_image_path)
person4_encoding = encode_face(person4_image_path)

known_face_encodings = [person1_encoding, person2_encoding, person3_encoding, person4_encoding]
known_face_names = ["Johnny Depp", "Johnny Depp", "Johnny Depp", "Amber Heard"]

# Process the video
input_video_path = 'deppheard.mp4'
output_video_path = '2stagefacerecog_facerecognition_yolo.mp4'
recognize_faces_in_video(input_video_path, output_video_path, known_face_encodings, known_face_names)

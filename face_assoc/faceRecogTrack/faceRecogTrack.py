import os
import torch
import dlib
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker


# Load face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


def detect_faces_yolo(frame, yolo):
    results = yolo(frame)
    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > 0.5:
                x,y = get_head_position([x1, y1, x2, y2])
                detections.append([x,y, score])
    return detections



def get_head_position(box):
    head_positions = []
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    head_y = int(y1 + 0.1 * box_height)  # Top 10% of the box
    head_x = int((x1 + x2) / 2)  # Center horizontally
    head_positions.append((head_x, head_y))
    return head_positions[0]


def load_database_images(database_path):
    database = {}
    for filename in os.listdir(database_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            label = os.path.splitext(filename)[0]  # Strip off extension for label
            img_path = os.path.join(database_path, filename)
            img = cv2.imread(img_path)
            database[label] = img
    return database

# Load database images
database_path = '../audio_face'
database = load_database_images(database_path)

def recognize_faces_dlib(frame, faces):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = [dlib.rectangle(x1, y1, x2, y2) for x1, y1, x2, y2 in faces]
    faces_encodings = []
    for rect in rects:
        shape = predictor(gray, rect)
        aligned_face = dlib.get_face_chip(frame, shape, size=160)  # Perform face alignment
        face_descriptor = face_rec_model.compute_face_descriptor(aligned_face)
        faces_encodings.append(face_descriptor)

    recognized_faces = []
    for face_encoding in faces_encodings:
        # Compare face_encoding with database
        min_distance = float('inf')
        recognized_label = None
        for label, encoding in database.items():
            distance = np.linalg.norm(np.array(encoding) - np.array(face_encoding))
            if distance < min_distance:
                min_distance = distance
                recognized_label = label

        if min_distance < 0.6:
            recognized_faces.append((recognized_label, True))
        else:
            recognized_faces.append(("Unknown", False))

    return recognized_faces

def process_video(video_path, output_csv='output.csv'):

    # Load YOLO model
    yolo = YOLO()

    cap = cv2.VideoCapture(video_path)

    # Initialize dictionary to track recognition status for each person
    recognition_status = {label: {'recognized_frames': [], 'unrecognized_frames': []} for label in database.keys()}
    person_coordinates = {}
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        detections = detect_faces_yolo(frame, yolo)
        faces = [(x1, y1, x2, y2) for x, y, _ in detections for x1, y1, x2, y2, _, _ in yolo.scale_boxes(frame.shape[:2], [x, y, x, y])]
        recognized_faces = recognize_faces_dlib(frame, faces)

        # Update recognition status for each person
        for (x1, y1, x2, y2), (label, recognized) in zip(faces, recognized_faces):
            if recognized:
                recognition_status[label]['recognized_frames'].append((frame_count, (x1, y1, x2, y2)))
                person_coordinates[label].append((frame_count, (x1, y1, x2, y2)))
            else:
                recognition_status[label]['unrecognized_frames'].append((frame_count, (x1, y1, x2, y2)))
                # Track faces using the tracker in unrecognized frames
                tracker.update(frame, [[x1, y1, x2, y2]])

            

        frame_count += 1

    

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Label', 'Track ID', 'Frame', 'Coordinates'])
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            # Check if the track corresponds to a person in the database
            for label, coords_list in person_coordinates.items():
                for frame_num, coords in coords_list:
                    if (x1, y1, x2, y2) == coords:
                        csv_writer.writerow([label, track_id, frame_num, coords])

        
        # for (x1, y1, x2, y2), (label, recognized) in zip(faces, recognized_faces):
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        # cv2.imshow('Frame', frame)


        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


    # cap.release()
    # cv2.destroyAllWindows()

    return recognition_status


vid_path = 'raghavchair.mp4'
process_video(vid_path)
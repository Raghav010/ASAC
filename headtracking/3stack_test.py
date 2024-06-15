import cv2
import face_recognition
from ultralytics import YOLO
import os
from tqdm import tqdm
import numpy as np
from yunetdetecttest import FaceDetectorYunet
from tracker import Tracker


# Function to encode a face from an image
def encode_face(image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    return encoding



def recognize_faces_in_frame(frame, yolomodel, yunetmodel, detection_threshold, known_face_encodings, known_face_names):


    # Colors
    face_color = (0, 255, 0)  # Green color for face rectangles
    yolo_color = (0, 0, 255)  # Red color for YOLO detections

    # a list of lists, each tuple is a face
    # list = [person_box, face_point, recognized person]
    # recognized person can be "Unknown"
    # if face box is not detected, then face point will be the top 33.33% of the person box
    face_info = []

    # Get person detections, YOLO detections
    results = yolomodel(frame, verbose=False)
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
                # face data
                face_data = [(x1, y1, x2, y2), ((x1+x2)/2, (2*y1+y2)/3), "Unknown"]
                face_info.append(face_data)


                detections.append([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (x1, y1), (x2, y2), yolo_color, 2)  # Red box for YOLO person, Draw rectangle for people detections

                # Get face detections from the person detection box
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                person_frame = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])

                # Get face locations using yunet
                face_boxes = yunetmodel.detect(person_frame)
                if not face_boxes:
                    continue

                # getting the max confidence face detection score
                confs = []
                for face_box in face_boxes:
                    confs.append(face_box['confidence'])
                
                # recognising the faces detected
                if len(confs) > 0:
                    fb = face_boxes[np.array(confs).argmax()]

                    face_data[1] = ((fb['x1']+fb['x2'])/2, (fb['y1']+fb['y2'])/2)

                    face_locations = [(fb['y1'], fb['x2'], fb['y2'], fb['x1'])]
                    face_encodings = face_recognition.face_encodings(person_frame, face_locations)

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = face_distances.argmin()
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        face_data[2] = name
                        cv2.rectangle(frame, (left+x1, top+y1), (right+x1, bottom+y1), face_color, 2)  # Green box for face
                        cv2.putText(frame, name, (left+x1 + 6, bottom+y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return frame, face_info




# Function to perform face recognition on a video
def recognize_faces_in_video(input_video_path, output_video_path, known_face_encodings, known_face_names):
    
    # parsing video and setting up video parameters
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

    # Initialize face detector
    fd = FaceDetectorYunet()

    # Detection parameters
    detection_threshold = 0.5

    # Colors
    face_color = (0, 255, 0)  # Green color for face rectangles
    yolo_color = (0, 0, 255)  # Red color for YOLO detections


    # Recognizing faces in each frame
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Get person detections, YOLO detections
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), yolo_color, 2)  # Red box for YOLO person, Draw rectangle for people detections

                    # Get face detections from the person detection box
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    person_frame = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])

                    # Get face locations using yunet
                    face_boxes = fd.detect(person_frame)
                    if not face_boxes:
                        continue

                    # getting the max confidence face detection score
                    confs = []
                    for face_box in face_boxes:
                        confs.append(face_box['confidence'])
                    
                    # recognising the faces detected
                    if len(confs) > 0:
                        fb = face_boxes[np.array(confs).argmax()]

                        face_locations = [(fb['y1'], fb['x2'], fb['y2'], fb['x1'])]
                        face_encodings = face_recognition.face_encodings(person_frame, face_locations)

                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            name = "Unknown"

                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = face_distances.argmin()
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]

                            cv2.rectangle(frame, (left+x1, top+y1), (right+x1, bottom+y1), face_color, 2)  # Green box for face
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
output_video_path = '3stack_yolo_yunet_facerecognition.mp4'

# parsing video and setting up video parameters
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

# Initialize face detector
fd = FaceDetectorYunet()


# initialize tracker
tracker = Tracker()

# Detection parameters
detection_threshold = 0.5

frames2faces = []

count=0
# Recognizing faces in each frame, and also tracking them
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    origframe = np.copy(frame)

    frame, face_info = recognize_faces_in_frame(frame, model, fd, detection_threshold, known_face_encodings, known_face_names)

    # tracking the faces
    dets = [[k[0][0],k[0][1],k[0][2],k[0][3],7] for k in face_info]
    tracker.update(origframe, dets)

    # updating the face_info with the track id
    print([track.bbox for track in tracker.tracks])
    print(face_info)
    print('\n\n\n')

    if count==5:
        break
    
    count+=1
    for track in tracker.tracks:
        for face in face_info:
            if tuple(track.bbox) == tuple(face[0]):
                face[-1] = track.track_id

    frames2faces.append(face_info)
    output_video.write(frame)
    pbar.update(1)


print(frames2faces[:10])

video_capture.release()
output_video.release()
pbar.close()







# recognize_faces_in_video(input_video_path, output_video_path, known_face_encodings, known_face_names)

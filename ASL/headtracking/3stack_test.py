import cv2
import face_recognition
from ultralytics import YOLO
import os
from tqdm import tqdm
import numpy as np
from yunetdetecttest import FaceDetectorYunet
from tracker import Tracker
import bisect


# Function to encode a face from an image
def encode_face(image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    return encoding


# returns the frame with people detected and a list of detections
# detections is in format [x1, y1, x2, y2, score]
def detect_people_in_frame(frame, yolomodel, detection_threshold, draw=True):
    # Colors
    yolo_color = (0, 0, 255)  # Red color for YOLO detections

    # Get person detections, YOLO detections
    results = yolomodel(frame, verbose=False)
    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold and class_id == 0:
                detections.append([x1, y1, x2, y2, score])
                if draw:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), yolo_color, 2)  # Red box for YOLO person, Draw rectangle for people detections

    return frame, detections



# returns the frame with faces detected and a list of detections
# It needs the bounding boxes of the people detected
# detections are in format [x1, y1, x2, y2]
# if no face was detected in a person box, then the face box will be the top 1/3rd of the person box
# to indicate that this was  default replacement, there will be a 0 in the 5th index of the face box
def detect_faces_in_frame(frame, yunetmodel, detection_threshold, bbox, draw=True):
    # Colors
    face_color = (0, 255, 0)  # Green color for face rectangles

    face_dets = []
    for bb in bbox:
        x1, y1, x2, y2 = bb
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        # by default the face detection is the top 1/3rd of the person box
        face_det = [x1, y1, x2, (y2+2*y1)/3, 0]
        face_dets.append(face_det)

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
            face_det.clear()
            face_det.extend([fb['x1'] + x1, fb['y1'] + y1, fb['x2'] +x1, fb['y2']+ y1])
        
        if draw:
            cv2.rectangle(frame, (face_det[0], face_det[1]), (face_det[2], face_det[3]), face_color, 2)  # Green box for face
        
    return frame, face_dets




# returns the frame with faces recognized and a list of names/recognized faces
# It needs the bounding boxes of the faces detected
# if face couldnt be recognized then the name will be "Unknown"
def recognize_faces_in_frame(frame, detection_threshold, face_bbox, known_face_encodings, known_face_names, draw=True):

    names = []

    for fb in face_bbox:
        if len(fb) == 0 or len(fb) == 5:
            names.append("Unknown")
            continue
        face_locations = [(fb[1], fb[2], fb[3], fb[0])]
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            names.append(name)
            if draw:
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            
    return frame, names




# old function
# def recognize_faces_in_frame(frame, yolomodel, yunetmodel, detection_threshold, known_face_encodings, known_face_names):


#     # Colors
#     face_color = (0, 255, 0)  # Green color for face rectangles
#     yolo_color = (0, 0, 255)  # Red color for YOLO detections

#     # a list of lists, each tuple is a face
#     # list = [person_box, face_point, recognized person]
#     # recognized person can be "Unknown"
#     # if face box is not detected, then face point will be the top 33.33% of the person box
#     face_info = []

#     # Get person detections, YOLO detections
#     results = yolomodel(frame, verbose=False)
#     for result in results:
#         detections = []
#         for r in result.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             x1 = int(x1)
#             x2 = int(x2)
#             y1 = int(y1)
#             y2 = int(y2)
#             class_id = int(class_id)
#             if score > detection_threshold and class_id == 0:
#                 # face data
#                 face_data = [(x1, y1, x2, y2), ((x1+x2)/2, (2*y1+y2)/3), "Unknown"]
#                 face_info.append(face_data)


#                 detections.append([x1, y1, x2, y2, score])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), yolo_color, 2)  # Red box for YOLO person, Draw rectangle for people detections

#                 # Get face detections from the person detection box
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 person_frame = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])

#                 # Get face locations using yunet
#                 face_boxes = yunetmodel.detect(person_frame)
#                 if not face_boxes:
#                     continue

#                 # getting the max confidence face detection score
#                 confs = []
#                 for face_box in face_boxes:
#                     confs.append(face_box['confidence'])
                
#                 # recognising the faces detected
#                 if len(confs) > 0:
#                     fb = face_boxes[np.array(confs).argmax()]

#                     face_data[1] = ((fb['x1']+fb['x2'])/2, (fb['y1']+fb['y2'])/2)

#                     face_locations = [(fb['y1'], fb['x2'], fb['y2'], fb['x1'])]
#                     face_encodings = face_recognition.face_encodings(person_frame, face_locations)

#                     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#                         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                         name = "Unknown"

#                         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#                         best_match_index = face_distances.argmin()
#                         if matches[best_match_index]:
#                             name = known_face_names[best_match_index]

#                         face_data[2] = name
#                         cv2.rectangle(frame, (left+x1, top+y1), (right+x1, bottom+y1), face_color, 2)  # Green box for face
#                         cv2.putText(frame, name, (left+x1 + 6, bottom+y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

#     return frame, face_info




# Function to perform face recognition on a video
# def recognize_faces_in_video(input_video_path, output_video_path, known_face_encodings, known_face_names):
    
#     # parsing video and setting up video parameters
#     video_capture = cv2.VideoCapture(input_video_path)
#     frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(video_capture.get(cv2.CAP_PROP_FPS))
#     total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     pbar = tqdm(total=total_frames, desc="Processing frames")

#     # Video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     # Initialize YOLO model
#     model = YOLO("yolov8n.pt")

#     # Initialize face detector
#     fd = FaceDetectorYunet()

#     # Detection parameters
#     detection_threshold = 0.5

#     # Colors
#     face_color = (0, 255, 0)  # Green color for face rectangles
#     yolo_color = (0, 0, 255)  # Red color for YOLO detections


#     # Recognizing faces in each frame
#     while video_capture.isOpened():
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         # Get person detections, YOLO detections
#         results = model(frame, verbose=False)
#         for result in results:
#             detections = []
#             for r in result.boxes.data.tolist():
#                 x1, y1, x2, y2, score, class_id = r
#                 x1 = int(x1)
#                 x2 = int(x2)
#                 y1 = int(y1)
#                 y2 = int(y2)
#                 class_id = int(class_id)
#                 if score > detection_threshold and class_id == 0:
#                     detections.append([x1, y1, x2, y2, score])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), yolo_color, 2)  # Red box for YOLO person, Draw rectangle for people detections

#                     # Get face detections from the person detection box
#                     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     person_frame = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])

#                     # Get face locations using yunet
#                     face_boxes = fd.detect(person_frame)
#                     if not face_boxes:
#                         continue

#                     # getting the max confidence face detection score
#                     confs = []
#                     for face_box in face_boxes:
#                         confs.append(face_box['confidence'])
                    
#                     # recognising the faces detected
#                     if len(confs) > 0:
#                         fb = face_boxes[np.array(confs).argmax()]

#                         face_locations = [(fb['y1'], fb['x2'], fb['y2'], fb['x1'])]
#                         face_encodings = face_recognition.face_encodings(person_frame, face_locations)

#                         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#                             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                             name = "Unknown"

#                             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#                             best_match_index = face_distances.argmin()
#                             if matches[best_match_index]:
#                                 name = known_face_names[best_match_index]

#                             cv2.rectangle(frame, (left+x1, top+y1), (right+x1, bottom+y1), face_color, 2)  # Green box for face
#                             cv2.putText(frame, name, (left+x1 + 6, bottom+y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

#         output_video.write(frame)
#         pbar.update(1)

#     video_capture.release()
#     output_video.release()
#     pbar.close()







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

# count=0
# Recognizing faces in each frame, and also tracking them
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # a list of lists
    # each list corresponds to a face
    # each face list looks like [person_box, track_id, face_point, names]
    faces = []
    origframe = np.copy(frame)

    frame, pboxs = detect_people_in_frame(frame, model, detection_threshold, False)

    # for pbox in pboxs:
    #     cv2.rectangle(frame, (int(pbox[0]), int(pbox[1]), int(pbox[2]), int(pbox[3])), (255, 0, 0), 2)

    # frame, face_info = recognize_faces_in_frame(frame, model, fd, detection_threshold, known_face_encodings, known_face_names)

    # tracking the faces
    # dets = [[k[0][0],k[0][1],k[0][2],k[0][3],7] for k in face_info]
    tracker.update(origframe, pboxs)

    # updating the face_info with the track id
    # print([track.bbox for track in tracker.tracks])
    # print(face_info)
    # print('\n\n\n')

    # if count==5:
    #     break
    
    # count+=1
    # for track in tracker.tracks:
    #     for face in face_info:
    #         if tuple(track.bbox) == tuple(face[0]):
    #             face[-1] = track.track_id
    if len(tracker.tracks) != 0:
        for track in tracker.tracks:
            # check if indexes are negative or if x1 and x2 are same or y1 and y2 are same
            if track.bbox[0] < 0 or track.bbox[1] < 0 or track.bbox[2] < 0 or track.bbox[3] < 0 or track.bbox[0] == track.bbox[2] or track.bbox[1] == track.bbox[3]:
                continue
            # cv2.rectangle(frame, (int(track.bbox[0]), int(track.bbox[1]), int(track.bbox[2]), int(track.bbox[3])), (0, 0, 255), 2)
            faces.append([tuple(track.bbox), track.track_id])
        
        # try:
        frame, fboxs = detect_faces_in_frame(frame, fd, detection_threshold, [face[0] for face in faces])
        # except:
        #     print(pboxs)
        #     print([tuple(track.bbox) for track in tracker.tracks])
        #     exit(0)
        for i in range(len(fboxs)):
            # faces[i].append(((fboxs[i][0]+fboxs[i][2])/2, (fboxs[i][1]+fboxs[i][3])/2))
            faces[i].append((fboxs[i][0], fboxs[i][1], fboxs[i][2], fboxs[i][3]))
        
        frame, names = recognize_faces_in_frame(frame, detection_threshold, fboxs, known_face_encodings, known_face_names)
        for i in range(len(faces)):
            faces[i].append(names[i])

    frames2faces.append(faces)
    output_video.write(frame)
    pbar.update(1)

video_capture.release()
output_video.release()
pbar.close()

# print(frames2faces[:10])


trackIds2frames = {}

# keep track of which frames track ids are present
for f2fidx in range(len(frames2faces)):
    for faceidx in range(len(frames2faces[f2fidx])):
        if frames2faces[f2fidx][faceidx][-1] == "Unknown":
            continue
        if frames2faces[f2fidx][faceidx][1] not in trackIds2frames:
            trackIds2frames[frames2faces[f2fidx][faceidx][1]] = []
        trackIds2frames[frames2faces[f2fidx][faceidx][1]].append((f2fidx, faceidx))

for trackid in trackIds2frames:
    trackIds2frames[trackid].sort(key=lambda x: x[0])

# print(trackIds2frames)

# try to identify all the unrecognized faces
for f2fidx in range(len(frames2faces)):
    for face in frames2faces[f2fidx]:
        if face[-1] == "Unknown":
            trackid = face[1]
            
            if trackid in trackIds2frames:
                # frames in which track id was recognized
                frames = trackIds2frames[trackid]

                # do binary search to find the closest frame to the current frame in which the track id was recognized
                highidx = bisect.bisect_left(frames, f2fidx, key = lambda x: x[0])

                if highidx == len(frames):
                    closest_frame = frames[-1]
                else:
                    diff = frames[highidx][0] - f2fidx
                    closest_frame = frames[highidx]
                    if highidx > 0:
                        diff2 = frames[highidx-1][0] - f2fidx
                        if diff2 < diff:
                            diff = diff2
                            closest_frame = frames[highidx-1]
                
                # get the name of the face in the closest frame
                face[-1] = frames2faces[closest_frame[0]][closest_frame[1]][-1]
                print("Identified face in frame", f2fidx, "as", face[-1])


# write another video using the information in frames2faces, the 2nd element which denotes the face, draw a rectange there and label it with the name


# Open the input video again
video_capture = cv2.VideoCapture(input_video_path)
output_video_path_labeled = '3stack_yolo_yunet_facerecognition_enhanced.mp4'

# Video writer
output_video_labeled = cv2.VideoWriter(output_video_path_labeled, fourcc, fps, (frame_width, frame_height))

# Process the frames and add labels
pbar = tqdm(total=total_frames, desc="Rendering enhanced version")
frame_index = 0

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    if frame_index < len(frames2faces):
        faces = frames2faces[frame_index]
        for face in faces:
            face_point = face[2]
            name = face[3]
            if len(face_point) == 0 or name == "Unknown":
                continue

            cv2.rectangle(frame, (int(face_point[0]), int(face_point[1]), int(face_point[2]), int(face_point[3])), (0, 255, 0), 2)
            cv2.putText(frame, name, (int(face_point[0]) + 6, int(face_point[3]) - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    output_video_labeled.write(frame)
    pbar.update(1)
    frame_index += 1

video_capture.release()
output_video_labeled.release()
pbar.close()





# recognize_faces_in_video(input_video_path, output_video_path, known_face_encodings, known_face_names)

import cv2
import face_recognition
from tqdm import tqdm
import numpy as np

# Load the known face image
known_image = face_recognition.load_image_file("audio_face/Johnny_Depp.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

known_image2 = face_recognition.load_image_file("audio_face/jd2.png")
known_encoding2 = face_recognition.face_encodings(known_image2)[0]

known_image3 = face_recognition.load_image_file("audio_face/jd3.png")
known_encoding3 = face_recognition.face_encodings(known_image3)[0]


known_image4 = face_recognition.load_image_file("audio_face/ah.png")
known_encoding4 = face_recognition.face_encodings(known_image4)[0]

known_encodings= [known_encoding, known_encoding2, known_encoding3, known_encoding4]
known_face_names = ["Johnny Depp","Johnny Depp2", "Johnny Depp3", "Amber Heard"]

vid_path = "deppheard.mp4"

# Open the video file for reading
video_capture = cv2.VideoCapture(vid_path)

# Get the total number of frames in the video
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(vid_path.split('.')[0]+'_output_dlib.avi', fourcc, fps, (frame_width, frame_height))

# Process each frame in the video
for _ in tqdm(range(total_frames)):
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB (required by face_recognition library)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize lists to store recognized and unrecognized face locations
    recognized_faces = []
    unrecognized_faces = []
    recognized_names = []

    for encoding, location in zip(face_encodings, face_locations):
        # Compare the current face encoding with the known encoding
        matches = face_recognition.compare_faces(known_encodings, encoding)

        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            recognized_names.append(name)
            recognized_faces.append(location)
        else:
            unrecognized_faces.append(location)

        


    # Draw green bounding boxes around recognized faces and red boxes around unrecognized faces
    for idx, (top, right, bottom, left) in enumerate(recognized_faces):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, recognized_names[idx], (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for top, right, bottom, left in unrecognized_faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, "unknown", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Write the processed frame to the output video
    output_video.write(frame)

    # # Display the processed frame
    # cv2.imshow('Video', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the video capture and writer objects
video_capture.release()
output_video.release()
cv2.destroyAllWindows()

print("\nProcessing complete")

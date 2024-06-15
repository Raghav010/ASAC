import cv2
import face_recognition

def encode_face(image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    return encoding

def recognize_faces_in_video(input_video_path, output_video_path, known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(input_video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        print(face_locations)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        output_video.write(frame)
    
    video_capture.release()
    output_video.release()

# Encode faces of two people
person1_image_path = '../face_assoc/audio_face/Johnny_Depp.jpg'
person2_image_path = '../face_assoc/audio_face/jd2.png'
person3_image_path = '../face_assoc/audio_face/jd3.png'
person4_image_path = '../face_assoc/audio_face/ah.png'
person1_encoding = encode_face(person1_image_path)
person2_encoding = encode_face(person2_image_path)
person3_encoding = encode_face(person3_image_path)
person4_encoding = encode_face(person4_image_path)

known_face_encodings = [person1_encoding, person2_encoding, person3_encoding, person4_encoding]
known_face_names = ["Johnny Depp", "Johnny Depp","Johnny Depp", "Amber Heard"]

# Process the video
input_video_path = 'deppheard.mp4'
output_video_path = 'facerecognition_recog_test.mp4'
recognize_faces_in_video(input_video_path, output_video_path, known_face_encodings, known_face_names)

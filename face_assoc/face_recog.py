import cv2
from deepface import DeepFace
from tqdm import tqdm
import csv

video_path = 'jd.mp4'
audio_repo = 'audio_face'
output_video_path = 'output_video.avi'  # Path to save the output video

cap = cv2.VideoCapture(video_path)

# get video meta data
fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('Video Properties')
print(fps, length, width, height)

coords = {}

face_recog_models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

face_det_models = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def draw_bounding_box(frame, face, recognized, aud_file=None, reason=None):
    x, y, w, h = int(face['facial_area']['x']), int(face['facial_area']['y']), \
                 int(face['facial_area']['w']), int(face['facial_area']['h'])
    color = (0, 255, 0) if recognized else (0, 0, 255)
    label = aud_file if recognized else reason
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

for framen in tqdm(range(length)):
    ret, frame = cap.read()
    if ret == True:
        # detect faces
        faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)

        if len(faces) == 0:
            # print('No face detected')
            continue

        # face recognition
        for face in faces:
            if face['confidence'] < 0.90:
                # print('Face confidence too low')
                draw_bounding_box(frame, face, False, reason='Confidence too low')
            else:
                df = DeepFace.find(img_path=face['face'], db_path=audio_repo, enforce_detection=False, silent=True,
                                   model_name="VGG-Face")[0]
                if len(df) == 0:
                    # print('No face recognised')
                    draw_bounding_box(frame, face, False, reason='No face recognised')
                else:
                    df.sort_values('distance', inplace=True)
                    if df['distance'][0] > df['threshold'][0]:
                        # print('No face recognised - Distance too high')
                        draw_bounding_box(frame, face, False, reason='Distance too high')
                    else:
                        aud_file = df['identity'][0].split('.')[0] + '.wav'
                        draw_bounding_box(frame, face, True, aud_file)

        # save the coordinates
        for aud_file in coords:
            if aud_file in rfaces:
                coords[aud_file].extend(rfaces[aud_file])
            else:
                coords[aud_file].extend([0, 0])

        # write the frame with bounding boxes to the output video
        out.write(frame)

    else:
        break

# write data to a csv file
with open('coordData.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([fps])
    for aud_file in coords:
        writer.writerow([aud_file] + coords[aud_file])

cap.release()
out.release()
cv2.destroyAllWindows()


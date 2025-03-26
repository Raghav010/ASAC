import cv2
from deepface import DeepFace
import shutil
import os
from tqdm import tqdm
import csv

video_path = 'jd.mp4'
audio_repo = 'audio_face'

cap = cv2.VideoCapture(video_path)

# get video meta data
fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('Video Properties')
print(fps, length, width, height)

# # create the directory
# if os.path.isdir(video_path.split('.')[0]):
#     shutil.rmtree(video_path.split('.')[0])
# os.mkdir(video_path.split('.')[0])

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

coords={}


for framen in tqdm(range(300)):
    ret, frame = cap.read()
    if ret==True:
        # detect faces
        faces = DeepFace.extract_faces(img_path = frame, enforce_detection=False)

        if len(faces)==0:
            print('No face detected')
            
        
        # recognised faces
        rfaces={}

        # face recognition
        for face in faces:
            if face['confidence']<0.90:
                print('Confidence too low')
                continue
            df = DeepFace.find(img_path = face['face'], db_path = audio_repo, enforce_detection=False, silent=True, model_name="GhostFaceNet")[0]
            if len(df)==0:
                print('No face recognised')
                continue
            df.sort_values('distance', inplace=True)
            if df['distance'][0]>df['threshold'][0]:
                print('No face recognised - Distance too high')
                continue
            aud_file = df['identity'][0].split('.')[0]+'.wav'
            if aud_file not in coords:
                coords[aud_file] = [0,0]*(framen)
            rfaces[aud_file] = [face['facial_area']['x']/width, face['facial_area']['y']/height]

        # save the coordinates
        for aud_file in coords:
            if aud_file in rfaces:
                coords[aud_file].extend(rfaces[aud_file])
            else:
                coords[aud_file].extend([0,0])

        # save the frame into a file
        # cv2.imwrite(video_path.split('.')[0] + '/' + str(framen) + '.jpg', frame)
    else:
        break


# write data to a csv file
with open('coordData.csv','w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([fps])
    for aud_file in coords:
        writer.writerow([aud_file]+coords[aud_file])


cap.release()
cv2.destroyAllWindows()


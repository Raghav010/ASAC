from moviepy.editor import VideoFileClip, concatenate_videoclips
from time import sleep
import subprocess


full_video = r"audiovid.mp4"
maxFrames = 250
mediaDir = 'media'
coords_csv = 'coords.csv'
speaker_audio_path = 'speaker_audio.wav'


clip = VideoFileClip(mediaDir+'/'+full_video)
fullDura = clip.duration
fps = int(clip.fps)
sampleRate = int(clip.audio.fps)
print(fps)
print(sampleRate)
startPos = 0

partDura = maxFrames/fps
print(partDura)

i = 0 
partNames = []
while True:
    endPos = startPos + partDura

    if endPos > fullDura:
        endPos = fullDura

    clip = VideoFileClip(mediaDir+'/'+full_video).subclip(startPos, endPos)

    part_name = f"{full_video.split('.')[0]}part_{str(i)}.mp4"
    partNames.append(part_name)
    clip.to_videofile(mediaDir + '/'+ part_name, codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')
    print("part ",i,"done")
    i += 1
    
    startPos = endPos # jump to next clip
    
    if (startPos >= fullDura) or (fullDura-startPos)<=(7/fps):
        break


# Call the command using subprocess
print('Running lwtnet')
command = f"python main.py --resume checkpoints/avobjects_loc_sep.pt --input_video {' '.join(partNames)} --output_dir {full_video.split('.')[0]} --coords_csv {coords_csv} --fps {25} --speaker_audio {speaker_audio_path} --sample_rate {16000}"
subprocess.run(command, shell=True)


# combining all video clips into one
# Assuming the output directory is where the individual videos are saved
print('Combining...')
output_dir = f"{full_video.split('.')[0]}/avobject_viz/"


video_files = [output_dir + f"{i}.mp4" for i in range(len(partNames))]
video_clips = [VideoFileClip(file) for file in video_files]
final_clip = concatenate_videoclips(video_clips)
combined_video_path = f"{full_video.split('.')[0]}_combined.mp4"
final_clip.to_videofile(combined_video_path, codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')

# Close all video clips to release resources
final_clip.close()
for clip in video_clips:
    clip.close()

print("Combined video created successfully at:", combined_video_path)



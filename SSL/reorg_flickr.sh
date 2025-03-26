#!/bin/bash

# Create the new directory structure
mkdir -p frames
mkdir -p audio

# Count files before reorganization (exclude the frames and audio directories)
jpg_count_before=$(find . -type f -name "*.jpg" -not -path "./frames/*" -not -path "./audio/*" | wc -l)
wav_count_before=$(find . -type f -name "*.wav" -not -path "./frames/*" -not -path "./audio/*" | wc -l)
mp3_count_before=$(find . -type f \( -name "*.mp3" -o -name "*.mp4.mp3" \) -not -path "./frames/*" -not -path "./audio/*" | wc -l)



# Find all JPG files (excluding the frames and audio directories) and move to frames directory
find . -type f -name "*.jpg" -not -path "./frames/*" -not -path "./audio/*" -print0 | while IFS= read -r -d '' file; do
    # Extract just the filename without the path
    filename=$(basename "$file")
    
    # Copy the file to the new location, keeping original name
    cp "$file" "frames/$filename"
    
    echo "Copied $file to frames/$filename"
done

# Find all WAV files (excluding the frames and audio directories) and move to audio directory
find . -type f -name "*.wav" -not -path "./frames/*" -not -path "./audio/*" -print0 | while IFS= read -r -d '' file; do
    # Extract just the filename without the path
    filename=$(basename "$file")
    
    # Copy the file to the new location, keeping original name
    cp "$file" "audio/$filename"
    
    echo "Copied $file to audio/$filename"
done

# Optional: Also handle the MP3 files if needed (excluding the frames and audio directories)
find . -type f \( -name "*.mp3" -o -name "*.mp4.mp3" \) -not -path "./frames/*" -not -path "./audio/*" -print0 | while IFS= read -r -d '' file; do
    # Extract just the filename without the path
    filename=$(basename "$file")
    
    # Copy the file to the new location, keeping original name
    cp "$file" "audio/$filename"
    
    echo "Copied $file to audio/$filename"
done

# Count files after reorganization
jpg_count_after=$(find frames -type f -name "*.jpg" | wc -l)
wav_count_after=$(find audio -type f -name "*.wav" | wc -l)
mp3_count_after=$(find audio -type f \( -name "*.mp3" -o -name "*.mp4.mp3" \) | wc -l)


echo "------------------------------"
echo "Before reorganization:"
echo "JPG files: $jpg_count_before"
echo "WAV files: $wav_count_before"
echo "MP3 files: $mp3_count_before"
echo "------------------------------"


echo "------------------------------"
echo "After reorganization:"
echo "JPG files: $jpg_count_after"
echo "WAV files: $wav_count_after"
echo "MP3 files: $mp3_count_after"
echo "------------------------------"

# Verify file counts match
if [ $jpg_count_before -eq $jpg_count_after ] && [ $wav_count_before -eq $wav_count_after ] && [ $mp3_count_before -eq $mp3_count_after ]; then
    echo "✅ Verification successful: All files were transferred correctly."
else
    echo "❌ Verification failed: File count mismatch!"
    echo "Check the logs above for details."
fi

echo "Directory reorganization complete!"
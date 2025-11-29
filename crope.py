import cv2
import os
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_FOLDER = 'facial data' # Your current folder
DEST_FOLDER = 'Cropped_Data'       # New folder for clean images

# Create destination folder if it doesn't exist
if not os.path.exists(DEST_FOLDER):
    os.makedirs(DEST_FOLDER)

# Load Face Detector (Haar Cascade)
# This usually comes with OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

files = os.listdir(SOURCE_FOLDER)
count = 0
success_count = 0

print(f"Processing {len(files)} images...")

for file in files:
    # Skip non-image files
    if not (file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith('.jpeg')):
        continue
        
    img_path = os.path.join(SOURCE_FOLDER, file)
    img = cv2.imread(img_path)
    
    if img is None:
        continue

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    save_path = os.path.join(DEST_FOLDER, file)

    if len(faces) > 0:
        # Get the biggest face (in case there are multiple)
        # x, y, w, h
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Add a little margin around the face so we don't cut the chin/forehead
        margin = 0 # Keep it tight for emotion recognition
        
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = w + 2*margin
        h = h + 2*margin
        
        # Crop
        face_img = img[y:y+h, x:x+w]
        
        # Resize to standard size immediately to save space/time
        try:
            face_img = cv2.resize(face_img, (224, 224))
            cv2.imwrite(save_path, face_img)
            success_count += 1
        except:
            # If cropping failed (e.g., margin went out of bounds), save original
            cv2.imwrite(save_path, cv2.resize(img, (224, 224)))
    else:
        # If NO FACE detected, save the original (resized)
        # Better to have a bad image than no image
        resized = cv2.resize(img, (224, 224))
        cv2.imwrite(save_path, resized)

    count += 1
    if count % 100 == 0:
        print(f"Processed {count} images...")

print(f"Done! {success_count} images were cropped to faces.")
print(f"All images saved in: {os.path.abspath(DEST_FOLDER)}")
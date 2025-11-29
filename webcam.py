import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# CONFIGURATION
# ==========================================
# Make sure this points to the BEST model file
MODEL_PATH = 'emotion_model_final.h5' 

# Load the model
print("Loading model... (This might take a few seconds)")
classifier = load_model(MODEL_PATH)

# Load Face Detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion Labels (Must match your training order)
# Usually alphabetical: Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise
class_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start Webcam
cap = cv2.VideoCapture(0) # Try 0, 1, or -1 if camera doesn't open

print("Starting Webcam... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # Crop face
        roi_color = frame[y:y+h, x:x+w]
        
        # Preprocess for MobileNetV2 (224x224, RGB)
        roi = cv2.resize(roi_color, (224, 224))
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        roi = preprocess_input(roi)

        # Predict
        prediction = classifier.predict(roi, verbose=0)[0]
        label = class_labels[prediction.argmax()]
        confidence = np.max(prediction)

        # Display Label
        label_position = (x, y - 10)
        text = f"{label} ({int(confidence*100)}%)"
        
        # Color code the text
        if label == 'Happy': color = (0, 255, 0)     # Green
        elif label == 'Anger': color = (0, 0, 255)   # Red
        else: color = (255, 255, 255)                # White
            
        cv2.putText(frame, text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Emotion Detector', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
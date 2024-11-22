import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model('person3.h5')

# Initialize the webcam capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to open the webcam.")
    exit()

# Set frame rate and time interval
fps = 2
interval = 1 / fps
last_time = time.time()

# Input image size (assume the model expects 200x200 input)
INPUT_SIZE = (200, 200)

def preprocess_frame(frame):
    """
    Preprocess the image to match the model input.
    """
    img = cv2.resize(frame, INPUT_SIZE)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Initialize detection status
current_status = None
label = ""
color = (0, 0, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    if current_time - last_time >= interval:
        last_time = current_time

        # Preprocess the captured frame from the webcam
        processed_frame = preprocess_frame(frame)
        logits = model.predict(processed_frame)[0]

        # Convert logits to probabilities
        probs = tf.nn.softmax(logits).numpy()
        prob_person = probs[0]        # Probability of class 0 (person)
        prob_person_like = probs[1]   # Probability of class 1 (person-like)
        prob_no_person = probs[2]     # Probability of class 2 (no-person)

        # Output debug information
        print(f"Logits: {logits}, Probabilities: {probs}")
        print(f"Person Probability: {prob_person:.2f}, Person-like Probability: {prob_person_like:.2f}, No-Person Probability: {prob_no_person:.2f}")

        # Select class based on probabilities
        max_index = np.argmax(probs)
        if max_index == 0:
            label = "Person Detected"
            color = (0, 255, 0)
        elif max_index == 1:
            label = "Person-like Detected"
            color = (255, 165, 0)
        else:
            label = "No Person"
            color = (0, 0, 255)

    # Display detection result on the video stream
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the video stream
    cv2.imshow('Real-Time Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

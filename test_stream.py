import cv2
import numpy as np
from tensorflow.keras.models import load_model
from CNNModel import CNNModel

# Load the trained model
loaded_model = load_model("model/mobilenetv2.h5")
labels = ['crosswalk', 'no_entry', 'speed_limit', 'stop']
model = CNNModel(loaded_model, labels)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess frame
    resized = cv2.resize(frame, (64, 64))
    normalized = resized / 255.0

    # Predict
    label, confidence = model.predict(normalized)

    # Display prediction
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Traffic Sign Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

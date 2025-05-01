import os
import cv2
import numpy as np
from CNNModel import CNNModel
from tensorflow.keras.models import load_model

# Load model and labels
loaded_model = load_model("model/mobilenetv2.h5")
labels = ['crosswalk', 'no_entry', 'speed_limit', 'stop']
model = CNNModel(loaded_model, labels)

# Directory containing test images
test_dir = "test_images"

# Loop through all files in the test_images directory
for filename in os.listdir(test_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(test_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        # Preprocess image
        resized = cv2.resize(image, (64, 64))
        normalized = resized / 255.0
        normalized = np.expand_dims(normalized, axis=0)

        # Predict
        label, confidence = model.predict(normalized)

        # Display result
        print(f"{filename} → Predicted: {label} ({confidence:.2f})")
        cv2.imshow(f"{filename} - Predicted: {label}", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

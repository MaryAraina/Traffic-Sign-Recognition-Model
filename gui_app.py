import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from CNNModel import CNNModel
from tensorflow.keras.models import load_model

# Load model and labels
loaded_model = load_model("model/mobilenetv2.h5")
labels = ['crosswalk', 'no_entry', 'speed_limit', 'stop']
model = CNNModel(loaded_model, labels)

class WelcomeScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Welcome - Traffic Sign Recognition")
        self.root.geometry("500x300")
        self.root.configure(bg="#f0f0f0")

        welcome_label = tk.Label(root, text="Welcome to Traffic Sign Recognition", font=("Arial", 18, "bold"), bg="#f0f0f0")
        welcome_label.pack(pady=40)

        instructions = tk.Label(root, text="Click below to begin using the application.", font=("Arial", 12), bg="#f0f0f0")
        instructions.pack(pady=10)

        start_button = tk.Button(root, text="Enter App", font=("Arial", 12), command=self.open_main_app)
        start_button.pack(pady=20)

    def open_main_app(self):
        self.root.destroy()
        main_root = tk.Tk()
        TrafficSignGUI(main_root)
        main_root.mainloop()

class TrafficSignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        self.root.geometry("500x550")

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Prediction label
        self.prediction_text = tk.StringVar()
        self.prediction_label = tk.Label(root, textvariable=self.prediction_text, font=("Arial", 14))
        self.prediction_label.pack()

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Exit", command=root.quit).pack(side=tk.LEFT, padx=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(display_img)
            tk_image = ImageTk.PhotoImage(pil_image.resize((300, 300)))

            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image

            # Predict
            resized = cv2.resize(image, (64, 64)) / 255.0
            label, confidence = model.predict(resized)
            self.prediction_text.set(f"Prediction: {label} ({confidence:.2f})")

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.prediction_text.set("Webcam not accessible.")
            return

        self.prediction_text.set("Press 'q' in webcam window to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (64, 64)) / 255.0
            label, confidence = model.predict(resized)

            # Overlay prediction
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Live Webcam - Press 'q' to exit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Launch welcome screen
if __name__ == "__main__":
    root = tk.Tk()
    WelcomeScreen(root)
    root.mainloop()

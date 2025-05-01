import cv2

class FrameStreamer:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def start_stream(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (64, 64)) / 255.0
            label, conf = self.model.predict(resized)
            self.logger.log("live_stream", label, conf)
            cv2.putText(frame, f"{label} ({conf*100:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Traffic Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

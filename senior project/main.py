from CNNModel import CNNModel
from ImageHandler import ImageHandler
from FrameStreamer import FrameStreamer
from Logger import Logger
from ExportManager import ExportManager

def test_image():
    handler = ImageHandler()
    model = CNNModel()
    logger = Logger()
    img = handler.load_image("test_images/speed_limit_50.jpg")
    label, confidence = model.predict(img)
    logger.log("speed_limit_50.jpg", label, confidence)
    print(f"Prediction: {label} ({confidence*100:.2f}%)")

def test_stream():
    model = CNNModel()
    logger = Logger()
    streamer = FrameStreamer(model, logger)
    streamer.start_stream()

if __name__ == "__main__":
    # Uncomment to test image mode
    # test_image()
    test_stream()

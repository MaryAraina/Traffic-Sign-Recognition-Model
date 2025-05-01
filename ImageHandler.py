from PIL import Image
import numpy as np

class ImageHandler:
    def load_image(self, path, size=(64, 64)):
        img = Image.open(path).resize(size).convert("RGB")
        return np.array(img) / 255.0

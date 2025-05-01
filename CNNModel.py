import numpy as np

class CNNModel:
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

    def predict(self, image):
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)

        if predicted_index < len(self.labels):
            return self.labels[predicted_index], confidence
        else:
            return "Unknown", confidence

import tensorflow as tf
import numpy as np

class CNNModel:
    def __init__(self, model_path="model/mobilenetv2.h5"):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image_array):
        prediction = self.model.predict(np.expand_dims(image_array, axis=0))[0]
        label_index = np.argmax(prediction)
        confidence = prediction[label_index]
        return self.get_label(label_index), confidence

    def get_label(self, index):
        labels = ["Speed Limit 50", "Stop", "Yield", "No Entry"]
        return labels[index]

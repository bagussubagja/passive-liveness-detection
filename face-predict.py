import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

IMG_SIZE = (96, 96)

model = tf.keras.models.load_model("liveness_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    decision = "real" if prediction >= 0.5 else "fake"
    confidence = float(prediction if prediction >= 0.5 else 1 - prediction)

    print(f"Confidence: {confidence:.4f}, Decision: {decision}")
    return confidence, decision

if __name__ == "__main__":
    predict_image("test_images/sample1.jpg") # real
    # predict_image("test_images/sample2.jpg") # fake

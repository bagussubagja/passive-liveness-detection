import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from mtcnn import MTCNN
import cv2

IMG_SIZE = (96, 96)
interpreter = tf.lite.Interpreter(model_path="liveness_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_indices = {'fake': 0, 'real': 1}
idx_to_label = {v: k for k, v in class_indices.items()}

detector = MTCNN()

def predict_image_tflite(img_path, conf_thresh=0.95, min_size=40):
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return 0.0, "unknown"
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    faces = [
        f for f in faces
        if f['confidence'] > conf_thresh and f['box'][2] > min_size and f['box'][3] > min_size
    ]

    if len(faces) == 0:
        print("Face detected: NO")
        print("Confidence: 0.0000, Decision: unknown")
        return 0.0, "unknown"

    print(f"Face detected: YES ({len(faces)} faces)")
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = np.argmax(prediction)
    decision = idx_to_label[pred_idx]
    confidence = float(prediction[pred_idx])

    print(f"Confidence: {confidence:.4f}, Decision: {decision}")
    return confidence, decision

if __name__ == "__main__":
    predict_image_tflite("test_images/sample1.jpg")

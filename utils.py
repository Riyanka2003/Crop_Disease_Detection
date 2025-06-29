import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

CLASS_NAMES = [
    'Pepper_bell__Bacterial_spot',
    'Pepper_bell__healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

def model_predict(model, img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        preds = model.predict(x)[0]
        top_index = np.argmax(preds)
        confidence = preds[top_index]
        return f"{CLASS_NAMES[top_index]} ({confidence:.2%} confidence)"
    except Exception as e:
        return f"Prediction failed: {e}"
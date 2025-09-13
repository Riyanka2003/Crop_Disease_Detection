import tensorflow as tf

try:
    model = tf.keras.models.load_model("model/model.h5")
    model.summary()
    print("\n✅ Model loaded successfully!")
except Exception as e:
    print("\n❌ Failed to load model:")
    print(str(e))
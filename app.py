import os
import gradio as gr
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from utils import model_predict
import tensorflow as tf

UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.path.join("model", "model.h5")

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model once at startup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        if 'image' not in request.files:
            error = "No file part"
        else:
            file = request.files['image']
            if file.filename == '':
                error = "No selected file"
            elif not allowed_file(file.filename):
                error = "Invalid file type. Please upload a PNG, JPG, or JPEG image."
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                try:
                    if model is None:
                        error = "Model not loaded. Please ensure the model file exists."
                    else:
                        prediction = model_predict(model, filepath)
                except Exception as e:
                    error = f"Prediction failed: {e}"
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
    app.run(debug=True)

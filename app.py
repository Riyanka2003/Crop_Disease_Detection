import os
import requests
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from utils import model_predict

# --- CONFIGURATION ---
MODEL_PATH = 'model/model.h5'
MODEL_URL = 'https://github.com/Riyanka2003/Crop_Disease_Detection/raw/main/model/model.h5'
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- MODEL LOADING LOGIC ---
def get_model():
    """
    Checks if the model file exists and is valid.
    If it's missing or too small (LFS pointer file), it downloads the real file from GitHub.
    """
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        print("⚠️ Model file is missing or invalid (LFS pointer detected).")
        print(f"⬇️ Downloading model from: {MODEL_URL}")
        
        try:
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Download complete.")
            else:
                print(f"❌ Failed to download model. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return None

    print("🔄 Loading Keras model...")
    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"❌ Error loading Keras model: {e}")
        return None

# Load the model ONCE at startup
model = get_model()

# --- HELPER FUNCTIONS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- ROUTES ---
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
                    # Clean up the uploaded file after prediction
                    if os.path.exists(filepath):
                        os.remove(filepath)

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == "__main__":
    # This block is for local development only!
    # Render uses Gunicorn and will ignore this block.
    app.run(debug=True)

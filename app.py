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

# Global variable to store the model
model = None

def download_file(url, filename):
    """
    Helper to download a file with a progress print.
    """
    print(f"⬇️ Downloading from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ Download complete.")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

# --- ROBUST MODEL LOADING ---
def load_model_robust():
    """
    Aggressively tries to load the model.
    If loading fails for ANY reason, it deletes the file and re-downloads.
    """
    print("🔄 Initializing Model Loading...")
    
    # Attempt 1: Try to load existing file
    if os.path.exists(MODEL_PATH):
        try:
            print("🔄 Found model file. Attempting to load...")
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"⚠️ Existing model file is corrupt or invalid: {e}")
            print("🗑️ Deleting corrupt file...")
            os.remove(MODEL_PATH)
    else:
        print("⚠️ Model file not found on disk.")

    # Attempt 2: Download and load
    print("🔄 Starting fresh download sequence...")
    if download_file(MODEL_URL, MODEL_PATH):
        try:
            print("🔄 Attempting to load newly downloaded model...")
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"❌ Critical Error: Downloaded model is also invalid: {e}")
            return None
    else:
        print("❌ Could not download model.")
        return None

# Load model ONCE at startup
model = load_model_robust()
if model:
    print("✅✅✅ SERVER READY: Model loaded successfully!")
else:
    print("❌❌❌ SERVER WARNING: Running without model!")

# --- HELPER FUNCTIONS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Use the global model variable
    global model
    
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
                    # Retry loading if model is missing
                    if model is None:
                        print("⚠️ Model was None. Retrying load...")
                        model = load_model_robust()
                    
                    if model is None:
                        error = "Model could not be loaded. Please check logs."
                    else:
                        prediction = model_predict(model, filepath)
                except Exception as e:
                    error = f"Prediction failed: {e}"
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)

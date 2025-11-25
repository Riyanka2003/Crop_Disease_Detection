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

# --- ROBUST MODEL LOADING ---
def load_model_robust():
    """
    Downloads and loads the model. Returns the model object or None.
    """
    print("🔄 Checking model status...")
    
    # 1. Check if model exists and is valid (not a fake LFS pointer)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        print("⚠️ Model missing or invalid. Downloading from GitHub...")
        try:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH) # Delete bad file
            
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Download complete.")
            else:
                print(f"❌ Failed to download. Status: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    # 2. Load the model
    print("🔄 Loading Keras model from disk...")
    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Load model ONCE at startup
model = load_model_robust()

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

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

# --- ROBUST MODEL LOADING ---
def get_model():
    """
    Tries to load the model. If it fails (corrupt file), 
    it automatically deletes the bad file and downloads a fresh one.
    """
    model = None
    print("🔄 Attempting to load model...")
    
    try:
        # First attempt to load
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully from disk!")
    except Exception as e:
        print(f"⚠️ Load failed (File might be corrupt): {e}")
        print("⬇️ Downloading fresh model from GitHub...")
        
        try:
            # Force download
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH) # Delete corrupt file
                
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Download complete.")
                
                # Second attempt to load
                model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Model loaded successfully after download!")
            else:
                print(f"❌ Failed to download. Status: {response.status_code}")
        except Exception as e2:
            print(f"❌ Critical Error: Could not download or load model: {e2}")
            
    return model

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
                        # Attempt to reload if it failed initially
                        global model
                        model = get_model()
                    
                    if model is None:
                        error = "Model could not be loaded. Check server logs."
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

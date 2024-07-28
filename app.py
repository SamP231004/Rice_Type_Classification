import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import h5py
import numpy as np
import os
import requests
from flask import Flask, request, render_template
import cv2

app = Flask(__name__)

# Firebase model URL
MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/rice-type-classification.appspot.com/o/rice.h5?alt=media&token=34a3d3bd-3502-43ea-a6a6-9d7a2def3ac9"

def download_model(url):
    response = requests.get(url)
    model_path = '/tmp/rice.h5'  # Temporary location
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print(f"Model downloaded and saved to {model_path}")
    return model_path

# Download and load the model at runtime
model_path = download_model(MODEL_URL)
model = tf.keras.models.load_model(filepath=model_path, custom_objects={'KerasLayer': hub.KerasLayer})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def pred():
    return render_template('details.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)  # getting the current path i.e where app.py is present
        filepath = os.path.join(basepath, 'Data', 'val', f.filename)  # save the uploaded image to a folder
        f.save(filepath)
        
        a2 = cv2.imread(filepath)
        a2 = cv2.resize(a2, (224, 224))
        a2 = np.array(a2)
        a2 = a2 / 255
        a2 = np.expand_dims(a2, 0)

        pred = model.predict(a2)
        pred = pred.argmax()

        df_labels = {
            'arborio': 0,
            'basmati': 1,
            'ipsala': 2,
            'jasmine': 3,
            'karacadag': 4
        }

        prediction = None
        for i, j in df_labels.items():
            if pred == j:
                prediction = i
        
        return render_template('results.html', prediction_text=prediction)
        
if __name__ == "__main__":
    app.run(debug=True)
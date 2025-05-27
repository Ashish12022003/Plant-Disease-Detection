import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('models/best_model.h5')

# Get class indices from training data
class_indices = {
    'healthy': 0,
    'diseased': 1
}

def preprocess_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # Save the uploaded file temporarily
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        
        # Preprocess the image and make prediction
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image, verbose=0)
        
        # Get the predicted class and confidence
        predicted_class = 'healthy' if prediction[0][0] > 0.5 else 'diseased'
        confidence = float(prediction[0][0] if predicted_class == 'healthy' else prediction[0][1])
        result = 'Diseased' if predicted_class == 'healthy' else 'Healthy'
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return jsonify({
            'class': result,
            'confidence': confidence,
            'image_path': f'/static/uploads/{file.filename}'
        })

if __name__ == '__main__':
    app.run(debug=True) 
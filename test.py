import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def test_model():
    # Load the trained model
    model = load_model('models/best_model.h5')
    
    # Get a sample image from test directory
    test_dir = 'data/test'
    healthy_dir = os.path.join(test_dir, 'healthy')
    diseased_dir = os.path.join(test_dir, 'diseased')
    
    # Test on one healthy and one diseased image
    test_images = [
        (os.path.join(healthy_dir, os.listdir(healthy_dir)[0]), 'Healthy'),
        (os.path.join(diseased_dir, os.listdir(diseased_dir)[0]), 'Diseased')
    ]
    
    # Create a figure to display results
    plt.figure(figsize=(12, 6))
    
    for idx, (img_path, true_label) in enumerate(test_images):
        # Load and preprocess image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_label = 'Healthy' if prediction[0][1] > prediction[0][0] else 'Diseased'
        confidence = prediction[0][1] if predicted_label == 'Healthy' else prediction[0][0]
        
        # Display image and prediction
        plt.subplot(1, 2, idx + 1)
        plt.imshow(img)
        plt.title(f'True: {true_label}\nPredicted: {predicted_label}\nConfidence: {confidence:.2%}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()
    
    print("Test completed! Results saved as 'test_results.png'")

if __name__ == "__main__":
    test_model() 
# Plant Disease Detection System

This project implements a deep learning-based system for detecting plant diseases from images. It uses a CNN (Convolutional Neural Network) model trained on plant disease images.

## Project Structure
```
plant_disease/
├── data/
│   ├── train/         # Training images
│   └── test/          # Testing images
├── models/            # Saved model files
├── static/           # Static files for web interface
│   ├── css/
│   ├── js/
│   └── uploads/      # Temporary storage for uploaded images
├── templates/        # HTML templates
├── app.py           # Flask application
├── model.py         # CNN model definition
├── train.py         # Training script
└── requirements.txt  # Python dependencies
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Organize your dataset:
   - Place training images in `data/train/` directory
   - Place testing images in `data/test/` directory
   - Organize images in subdirectories by disease class

4. Train the model:
```bash
python train.py
```

5. Run the web application:
```bash
python app.py
```

## Data Organization
- Training images should be organized in subdirectories by disease class
- Example structure:
  ```
  data/
  ├── train/
  │   ├── healthy/
  │   ├── disease1/
  │   └── disease2/
  └── test/
      ├── healthy/
      ├── disease1/
      └── disease2/
  ```

## Usage
1. Access the web interface at `http://localhost:5000`
2. Upload a plant image
3. Get instant disease detection results 
import os
import requests
import zipfile
from tqdm import tqdm
import shutil
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw

def create_leaf_shape(size):
    """Create a leaf-like shape using bezier curves"""
    width, height = size
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Create leaf shape using bezier curves
    points = [
        (width//2, 0),  # top
        (width, height//4),  # right top
        (width, height//2),  # right middle
        (width//2, height),  # bottom
        (0, height//2),  # left middle
        (0, height//4),  # left top
    ]
    
    # Draw leaf shape
    draw.polygon(points, fill=(0, 0, 0, 255))
    return image

def add_texture(image, color, is_diseased):
    """Add texture to the leaf"""
    width, height = image.size
    texture = np.array(image)
    
    # Add base color
    base_color = np.array(color)
    if is_diseased:
        # For diseased leaves, add brown spots and patches
        spots = np.random.rand(height, width) < 0.3
        texture[spots] = [139, 69, 19, 255]  # Brown spots
        # Add some yellowing
        yellowing = np.random.rand(height, width) < 0.2
        texture[yellowing] = [255, 255, 0, 255]  # Yellow patches
    else:
        # For healthy leaves, add some natural variation
        variation = np.random.normal(0, 20, (height, width, 3))
        texture[:, :, :3] = np.clip(base_color[:3] + variation, 0, 255)
    
    return Image.fromarray(texture)

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def download_and_organize_data():
    """Download and organize the plant leaves dataset"""
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download the dataset
    print("Downloading dataset...")
    url = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    download_file(url, 'data/plant_disease_data.zip')
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile('data/plant_disease_data.zip', 'r') as zip_ref:
        zip_ref.extractall('data')
    
    # Create train and test directories
    for split in ['train', 'test']:
        os.makedirs(f'data/{split}', exist_ok=True)
    
    # Organize the data
    print("Organizing data...")
    source_dir = 'data/Plant_leave_diseases_dataset_without_augmentation'
    
    # Split ratio for training/test
    split_ratio = 0.8
    
    # Process each category
    for category_dir in os.listdir(source_dir):
        if category_dir == 'Background_without_leaves':
            continue
            
        # Create category directory in train and test
        os.makedirs(f'data/train/{category_dir}', exist_ok=True)
        os.makedirs(f'data/test/{category_dir}', exist_ok=True)
        
        # Get all images for this category
        images = list(Path(os.path.join(source_dir, category_dir)).glob('*.JPG'))
        random.shuffle(images)
        
        # Calculate split point
        split_point = int(len(images) * split_ratio)
        
        # Move training images
        for img in images[:split_point]:
            shutil.copy2(str(img), f'data/train/{category_dir}/{img.name}')
        
        # Move test images
        for img in images[split_point:]:
            shutil.copy2(str(img), f'data/test/{category_dir}/{img.name}')
    
    # Clean up downloaded file and temporary directory
    if os.path.exists('data/plant_disease_data.zip'):
        os.remove('data/plant_disease_data.zip')
    if os.path.exists('data/Plant_leave_diseases_dataset_without_augmentation'):
        shutil.rmtree('data/Plant_leave_diseases_dataset_without_augmentation')
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("Training Set:")
    for category in os.listdir('data/train'):
        print(f"  - {category}: {len(os.listdir(f'data/train/{category}'))} images")
    
    print("\nTest Set:")
    for category in os.listdir('data/test'):
        print(f"  - {category}: {len(os.listdir(f'data/test/{category}'))} images")

def reorganize_data():
    """Reorganize existing data into healthy and diseased folders"""
    # Create temporary directories for reorganization
    temp_train = 'data/temp_train'
    temp_test = 'data/temp_test'
    
    # Create all necessary directories
    for temp_dir in [temp_train, temp_test]:
        for category in ['healthy', 'diseased']:
            os.makedirs(os.path.join(temp_dir, category), exist_ok=True)
    
    # Process each category in the existing train directory
    source_dir = 'data/train'
    
    # Process each category
    for category_dir in os.listdir(source_dir):
        if category_dir in ['healthy', 'diseased', 'temp_train', 'temp_test']:
            continue
            
        # Determine if this is a healthy or diseased category
        is_healthy = 'healthy' in category_dir.lower()
        target_category = 'healthy' if is_healthy else 'diseased'
        
        # Get all images for this category
        images = list(Path(os.path.join(source_dir, category_dir)).glob('*.JPG'))
        if not images:  # Try lowercase extension
            images = list(Path(os.path.join(source_dir, category_dir)).glob('*.jpg'))
        
        random.shuffle(images)
        
        # Calculate split point (80% for train, 20% for test)
        split_point = int(len(images) * 0.8)
        
        print(f"Processing {category_dir}: {len(images)} images")
        
        # Move training images
        for img in images[:split_point]:
            shutil.copy2(str(img), os.path.join(temp_train, target_category, img.name))
        
        # Move test images
        for img in images[split_point:]:
            shutil.copy2(str(img), os.path.join(temp_test, target_category, img.name))
    
    # Remove old directories and move new ones into place
    shutil.rmtree('data/train', ignore_errors=True)
    shutil.rmtree('data/test', ignore_errors=True)
    os.rename(temp_train, 'data/train')
    os.rename(temp_test, 'data/test')
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("Training Set:")
    print(f"  - Healthy images: {len(os.listdir('data/train/healthy'))}")
    print(f"  - Diseased images: {len(os.listdir('data/train/diseased'))}")
    print("\nTest Set:")
    print(f"  - Healthy images: {len(os.listdir('data/test/healthy'))}")
    print(f"  - Diseased images: {len(os.listdir('data/test/diseased'))}")

if __name__ == "__main__":
    reorganize_data() 
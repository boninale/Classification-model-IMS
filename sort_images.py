import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms, models
from torchvision.transforms import v2
from PIL import Image, UnidentifiedImageError
import torch.nn as nn

from architectures import EffNetB0, TLregularizer
from preprocessing import preprocess_image

# Define image size and path to new data
IMG_SIZE = (224, 224)
new_data_path = r'C:\Users\Alexandre Bonin\Documents\Stage\datasets\ProspectFD\Minervois\CameraB\p0902_0857'
output_path = r'C:\Users\Alexandre Bonin\Documents\Stage\datasets\ProspectFD\Minervois\CameraB\p0902_0857\sorted_balanced'
num_classes = 2
model_path = 'models/Run_2024-10-16_14-15-17.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device, "/n")
model = EffNetB0(num_classes, model_path).to(device)

print(f'Model loaded from {model_path} \n')

# Load the trained model state dictionary
model.eval()

# Create directories for each class
class_names = ['turn','vine']
for class_name in class_names:
    os.makedirs(os.path.join(output_path, class_name), exist_ok=True)

print('loading images...')


# Get list of new images
new_images = [os.path.join(new_data_path, fname) for fname in os.listdir(new_data_path) if fname.endswith(('jpg', 'jpeg', 'png'))]

# Make predictions and store them
predictions = {}
with torch.no_grad():
    for image_path in tqdm(new_images, desc = 'Predicting') :
        img_tensor = preprocess_image(image_path)
        if img_tensor is not None:
            try:
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]
                predictions[image_path] = predicted_class
            except UnidentifiedImageError:
                print(f"Warning: Skipping corrupted image file {image_path}")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

print('Predictions complete!')

# # Display a sample of predictions
# for image_path in list(predictions.keys())[:5]:  # Display first 5 images
#     print(f'Image: {os.path.basename(image_path)}, Predicted Class: {predictions[image_path]}')

# Move images to the corresponding class folders
for image_path, predicted_class in tqdm(predictions.items(), desc="Moving images..."):
    shutil.copy(image_path, os.path.join(output_path, predicted_class, os.path.basename(image_path)))

print(f'Moved images to {output_path}')
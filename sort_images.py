import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms, models
from torchvision.transforms import v2
from PIL import Image, UnidentifiedImageError
from architectures import EffNetB0
import torch.nn as nn

# Define image size and path to new data
IMG_SIZE = (224, 224)
new_data_path = 'C:/Users/Alexandre Bonin/Documents/Stage/datasets/ProspectFD/Reparsac-machine-20211007/p1007_0825'
output_path = 'C:/Users/Alexandre Bonin/Documents/Stage/datasets/ProspectFD/Reparsac-machine-20211007/p1007_0825/sorted'
num_classes = 3
model_path = 'models/Run_2024-10-15_09-11-23.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device, "/n")
model = EffNetB0(num_classes, model_path).to(device)

print(f'Model loaded from {model_path} \n')

# Load the trained model state dictionary
model.eval()

# Create directories for each class
class_names = ['missing_vine', 'turn','vine']
for class_name in class_names:
    os.makedirs(os.path.join(output_path, class_name), exist_ok=True)

print('loading images...')

# Define the image preprocessing function
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess new images
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(device)
        return img_tensor
    except UnidentifiedImageError:
        print(f"Skipping corrupted image: {image_path}")
        return None

# Get list of new images
new_images = [os.path.join(new_data_path, fname) for fname in os.listdir(new_data_path) if fname.endswith(('jpg', 'jpeg', 'png'))]

print('Predicting...')

# Make predictions and store them
predictions = {}
with torch.no_grad():
    for image_path in tqdm(new_images, desc = 'Predicting') :
        img_tensor = preprocess_image(image_path)
        if img_tensor is not None :
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]
            predictions[image_path] = predicted_class

print('Predictions complete!')

# Display a sample of predictions
for image_path in list(predictions.keys())[:5]:  # Display first 5 images
    print(f'Image: {os.path.basename(image_path)}, Predicted Class: {predictions[image_path]}')

print('Moving images...')
# Move images to the corresponding class folders
for image_path, predicted_class in predictions.items():
    shutil.copy(image_path, os.path.join(output_path, predicted_class, os.path.basename(image_path)))
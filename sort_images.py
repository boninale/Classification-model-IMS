import os
import shutil
import numpy as np
import torch
from torchvision import transforms, models
from torchvision.transforms import v2
from PIL import Image
import torch.nn as nn

# Define the model architecture
def create_model():
    """Create and compile the model."""
    model = models.efficientnet_b0(weights=None)  # Initialize model without pre-trained weights
    
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3),
        nn.Softmax(dim=1)
    )
    
    return model

# Create the model instance
model = create_model()

# Load the trained model state dictionary
model.load_state_dict(torch.load('models/EffNetB0_classifier_2.pth', map_location=torch.device('cpu')))
model.eval()


# Define image size and path to new data
IMG_SIZE = (224, 224)
new_data_path = 'datasets/test'
output_path = 'datasets/sorted'

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
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

# Get list of new images
new_images = [os.path.join(new_data_path, fname) for fname in os.listdir(new_data_path) if fname.endswith(('jpg', 'jpeg', 'png'))]

print('Predicting...')

# Make predictions and store them
predictions = {}
with torch.no_grad():
    for image_path in new_images:
        img_tensor = preprocess_image(image_path)
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
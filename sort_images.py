import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms, models
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError
import torch.nn as nn

from utils.architectures import EffNetB0, TLregularizer
from utils.preprocessing import ImageDataset, custom_collate
if __name__ == '__main__':
    # Define image size and path to new data
    IMG_SIZE = (224, 224)
    new_data_path = r'C:\Users\Alexandre Bonin\Documents\Stage\datasets\Luchey\p0423_1124'
    output_path = r'C:\Users\Alexandre Bonin\Documents\Stage\datasets\Luchey\p0423_1124\sorted'
    num_classes = 2
    model_path = 'models/Run_2024-11-08_10-18-59.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "\n")

    model = EffNetB0(num_classes, model_path).to(device)

    print(f'Model loaded from {model_path} \n')

    # Load the trained model state dictionary
    model.eval()

    # Create directories for each class
    class_names = ['accepted','rejected']
    for class_name in class_names:
        os.makedirs(os.path.join(output_path, class_name), exist_ok=True)

    print('loading images...')

    # Get list of new images
    new_images = [fname for fname in os.listdir(new_data_path)[:500] if fname.endswith(('jpg', 'jpeg', 'png'))]

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Batch size for predictions
    batch_size = 64 if torch.cuda.is_available() else 8
    num_workers = 8 if torch.cuda.is_available() else 0

    # Create dataset and dataloader
    dataset = ImageDataset(new_images, new_data_path, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=4)

    # Make predictions and store them
    predictions = {}
    with torch.no_grad():
        for img_names, img_tensors in tqdm(dataloader, desc='Predicting'):
            if img_tensors is not None:
                try:
                    outputs = model(img_tensors)
                    _, predicted = torch.max(outputs, 1)
                    for img_name, pred in zip(img_names, predicted):
                        predicted_class = class_names[pred.item()]
                        predictions[os.path.join(new_data_path, img_name)] = predicted_class
                except Exception as e:
                    for img_name in img_names:
                        print(f"Error processing image {img_name}: {e}")

    print('Predictions complete!')

    # Move images to the corresponding class folders
    for image_path, predicted_class in tqdm(predictions.items(), desc="Moving images..."):
        shutil.copy(image_path, os.path.join(output_path, predicted_class, os.path.basename(image_path)))

    print(f'Moved images to {output_path}')
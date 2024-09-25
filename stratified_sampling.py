import os
import random
import shutil


# Define the path to the classification folder
classification_folder = 'datasets/classification_balanced'
new_dataset_folder = 'datasets/test'

# Get a list of all subfolders in the classification folder
subfolders = [f.path for f in os.scandir(classification_folder) if f.is_dir()]

# Find the maximum number of photos in a folder
min_photos = min(len(os.listdir(subfolder)) for subfolder in subfolders)

# Calculate the target number of photos per folder with a 10% margin
target_photos = 10
print(f'Target number of photos per folder: {target_photos}')

# Iterate over each subfolder
for subfolder in subfolders:
    # Get the list of photos in the subfolder
    photos = os.listdir(subfolder)
    
    if len(photos) >= target_photos:
        
    # Randomly sample photos to match the target number
        sampled_photos = random.sample(photos, target_photos)
        
        # Create the corresponding subfolder in the new dataset folder
        new_subfolder = os.path.join(new_dataset_folder, os.path.basename(subfolder))
        if not os.path.exists(new_subfolder):
            os.makedirs(new_subfolder)
        
        # Copy the sampled photos to the new subfolder
        for photo in sampled_photos:
            src_path = os.path.join(subfolder, photo)
            dst_path = os.path.join(new_subfolder, photo)
            shutil.copy(src_path, dst_path)
        
        print(f'{new_subfolder} completed.')
    else:
        shutil.copytree(subfolder, os.path.join(new_dataset_folder, os.path.basename(subfolder)))
        print(f'Copied {new_subfolder}')

print('Stratified sampling completed.')
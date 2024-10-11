import os
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process some strings.')

# Add arguments
parser.add_argument('--folder_path', '-f', type=str, default='', help='Path to the folder')
parser.add_argument('--suffix', '-s', type=str, default='', help='Suffix for the files')
parser.add_argument('--prefix', '-p', type=str, default='', help='Prefix for the files')

# Parse the arguments
args = parser.parse_args()

# Update variables with parsed arguments

folder_path = 'C:/Users/Alexandre Bonin/Documents/Stage/datasets/ProspectFD/Minervois/CameraA/p0901_1433/p0901_1433_A/vine'
suffix = args.suffix
prefix = 'p0901_1433_A_' #args.prefix

def validate_arguments(folder_path, prefix, suffix):
    if not folder_path:
        raise ValueError("The folder path cannot be empty.")
    if not prefix and not suffix:
        raise ValueError("Both prefix and suffix cannot be empty.")


def append_prefix_to_files(folder_path, prefix, suffix):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Construct the full file path
        old_file_path = os.path.join(folder_path, filename)
        
        # Skip directories
        if os.path.isdir(old_file_path):
            continue
        
        # Split the filename into name and extension
        name, ext = os.path.splitext(filename)
    
        # Construct the new file name with the prefix and suffix
        new_filename = prefix + name + suffix + ext
        new_file_path = os.path.join(folder_path, new_filename)
    
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {old_file_path} to {new_file_path}')

# Validate arguments
validate_arguments(folder_path, prefix, suffix)
append_prefix_to_files(folder_path, prefix, suffix)
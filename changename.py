import os

def append_prefix_to_files(folder_path, prefix):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Construct the full file path
        old_file_path = os.path.join(folder_path, filename)
        
        # Skip directories
        if os.path.isdir(old_file_path):
            continue
        
        # Construct the new file name with the prefix
        new_filename = prefix + filename
        new_file_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {old_file_path} to {new_file_path}')

# Example usage
folder_path = 'datasets/p0720_0906'
prefix = 'p0720_0906_'
append_prefix_to_files(folder_path, prefix)
import torch

# Load the saved state_dict
state_dict = torch.load('C:/Users/Alexandre Bonin/Documents/Stage/Classification-model-IMS/models/Run_2024-10-15_14-06-31.pth', map_location = torch.device('cpu'))

# Remove 'model.' prefix from the keys
new_state_dict = {}
for key in state_dict.keys():
    # # Remove 'model.' from the key
    # new_key = key.replace('model.', '')
    # new_state_dict[new_key] = state_dict[key]

    # Add 'model.' to the key
    new_key = 'model.' + key
    new_state_dict[new_key] = state_dict[key]

# Save the new state dict
torch.save(new_state_dict, 'C:/Users/Alexandre Bonin/Documents/Stage/Classification-model-IMS/models/Run_2024-10-15_14-06-31-corrected.pth')

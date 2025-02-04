# Classification-model-IMS

This repository provides a comprehensive pipeline for training and deploying a classification model to sort images into specific classes. It is designed to be straightforward to set up, use, and replicate by other teams. Below is an overview of each file and its primary function, followed by general usage guidelines.

---

## Repository Structure

1. **README.md**  
   - Overview of the repository and instructions on how to use the codebase.

2. **utils/preprocessing.py**  
   - Defines the dataset classes and methods for splitting and creating data loaders.  
   - Contains the custom collate function to handle broken or missing images.  
   - Includes functions to preprocess images for inference or during model training.

3. **utils/changestatedict.py**  
   - Demonstrates how to manipulate the keys in a saved PyTorch state dictionary (e.g., adding or removing prefixes).  
   - Example usage: adjusting the model checkpoint to fit different naming conventions.

4. **utils/architectures.py**  
   - Contains custom model classes (e.g., EfficientNet-based classifier, triplet-loss regularizer, and a custom UNet).  
   - Shows how to modify or extend pretrained models with custom heads for classification and embedding outputs.

5. **sort_images.py**  
   - Main script for loading a trained model and classifying new images.  
   - Automatically sorts the images into folders named after the predicted classes.  
   - Accepts user-defined paths for both the input data and output directories.

6. **classifier_train.py**  
    - This script sets a reproducible environment by calling `set_seed` and defines data loaders (train/validation) via `create_data_generators`.
    - It configures two learning-rate schedulers: `ExponentialLR` and `ReduceLROnPlateau`.
    - The `train_model` function handles mixed-precision training, logs metrics (loss, accuracy, recall), and     applies early stopping based on validation loss.
    - After initial training, it fine-tunes by unfreezing parameters and reducing the learning rate.
    - It logs outputs with MLflow, including metrics, tags, and the trained model (via `mlflow.pytorch.log_model`).
    - Model weights (.pth) and MLflow artifacts are saved to ensure reproducibility and easy reloading. 
---

## Installation and Environment Setup

1. **Clone the Repository**  
   ```bash
   git clone git@gitub.u-bordeaux.fr:imagro/classification-model.git
   ```

2. **Install Dependencies**  
   - Recommended: Create and activate a virtual environment (e.g., using `venv` or `conda`).  
   - Install the necessary Python packages (collect requirements from the scripts or a `requirements.txt` if provided):  
   ```bash
   pip install -r requirements.txt
   ```

3. **Check GPU Availability**  
   - The scripts automatically detect a GPU if available (CUDA). Otherwise, they fall back to the CPU.

---

## Data Preparation

- Organize your image folders under a main directory with subfolders for each class (e.g., `train/dog`, `train/cat` for a binary classification).  
- Update the paths in each script (e.g., `datapath`, `new_data_path`) according to your file structure.

---

## Training & Validation

1. **Adjust Training Script**  
   - In `classifier_train.py`, set:  
     - `DATAPATH` to point to your training images directory.  
     - (Optional) `VAL_PATH` if you maintain a separate validation set.  
   - Choose the model type (`EffNetB0` or `TLregularizer`) based on your needs, and make sure to set `num_classes` correctly.

2. **Run the Training Script**  
   - This includes data loading, augmentation, and model logging with MLflow.  
   - Example command:
   ```bash
   python classifier_train.py
   ```

3. **Monitor Progress**  
   - Training logs (loss, accuracy, etc.) may be recorded via MLflow or printed to the console.  
   - The best model weights will be saved in the specified path. 
   - After running the script, start the MLflow UI to visualize metrics, parameters, and model artifacts. In bash, run in the training directory :
    ```bash
    mlflow ui -–host 0.0.0.0 -–port 5000
    ```
    Access the dashboard at  `http://<ip address of the server>:5000`.  (The ip address of the deepwater server is 192.168.215.58)


---

## Inference and Sorting Images

1. **Model Checkpoint**  
   - Copy your saved model `.pth` file into `models/` or another configured folder.  
   - Adjust the `model_path` in `sort_images.py` to point to your checkpoint.

2. **Run `sort_images.py`**  
   - Updates:  
     - `new_data_path` with your input dataset directory.  
     - `output_path` with the desired folder for sorted images.  
     - `num_classes` and class names for your use case.  
   - Example command:
   ```bash
   python sort_images.py
   ```

3. **Output**  
   - The script will create sub-folders under the `output_path` according to the class names you gave in the code, moving or copying images into each predicted class folder.

---

## Utilities

- **preprocessing.py**  
  - `create_training_generators(...)` and `create_data_generators(...)` handle transforms and splitting.  
  - `ImageDataset` and `custom_collate` manage dataset structure and safe loading of images.

- **changestatedict.py**  
  - Modifies checkpoint keys by removing or adding a prexif to them. Used when changed model definition and needed to adapt older models.

- **architectures.py**  
  - Easily extendable for different model architectures or custom classification heads.

---

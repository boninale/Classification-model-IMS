import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import time
from datetime import datetime
from sklearn.metrics import recall_score


def set_seed(seed=31415):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_matplotlib():
    """Set up Matplotlib defaults."""
    plt.style.use('ggplot')
    plt.rc('figure', autolayout=True)
    plt.rc('axes', labelweight='bold', labelsize='large',
           titleweight='bold', titlesize=18, titlepad=10)
    plt.rc('animation', html='html5')
    plt.rc('image', cmap='magma')
    rcParams['figure.figsize'] = (18, 8)
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

def create_data_generators(datapath, img_size, batch_size=25, val_split=0.2):
    """Create training and validation data generators."""
    transform = transforms.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(root=datapath, transform=transform)
    
    # Calculate the number of samples for training and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def create_model(img_size):
    """Create and compile the model."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3),
        nn.Softmax(dim=1)
    )
    
    return model

def get_callbacks(savepath):
    """Create callbacks for training."""
    early_stopping = {
        'patience': 10,
        'min_delta': 0.001,
        'counter': 0,
        'best_loss': None,
        'early_stop': False
    }

    checkpoint_path = os.path.join(savepath, f'{run_name}.pth')

    return early_stopping, checkpoint_path

def plot_history(history_df, save=True):
    """Plot training and validation loss."""
    history_df.loc[:, ['loss', 'val_loss']].plot()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if save:
        existing_files = [f for f in os.listdir(graphs_folder) if f.startswith('training_validation_loss') and f.endswith('.png')]
        file_number = len(existing_files) + 1
        save_path = os.path.join(graphs_folder, f'training_validation_loss_{file_number}.png')
        plt.savefig(save_path)

    plt.show()

    print("\n", "Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()), "\n")
    print("Corresponding Validation Accuracy: {:0.4f}".format(history_df['val_accuracy'][history_df['val_loss'].idxmin()]), "\n")

def example_data_loader(img_size=(224, 224)):
    # Assuming typical input is a batch of images from the data loader
    train_loader, _ = create_data_generators(DATAPATH, img_size)
    example_input, _ = next(iter(train_loader))
    example_input = example_input.to(device)
    example_output = model(example_input)

    return example_input, example_output

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, callbacks, device):
    """Train the model."""
    best_model = model
    # Log model parameters (e.g., hyperparameters)
    mlflow.log_param("learning_rate", optimizer.defaults['lr'])
    mlflow.log_param("batch_size", train_loader.batch_size)
    mlflow.log_param("epochs", epochs)

    # Set some tags for metadata
    
    mlflow.set_tag("model_type", "EfficientNetB0")
    #mlflow.set_tag("note", "First run using MLflow")

    total_training_time = 0  # Track total training time for all epochs
    total_steps = 0  # Track total steps across all epochs

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(epochs):

        start_time = time.time()  # Start timing the epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        end_time = time.time()  # End timing the epoch
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        #### MODEL EVALUATION ####
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_predictions = []
    
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
    
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Calculate recall for each class
        recall_per_class = recall_score(all_labels, all_predictions, average=None)
        for i, recall in enumerate(recall_per_class):
            mlflow.log_metric(f"val_recall_{class_names[i]}", recall, step=epoch)
         
        epoch_training_time = end_time - start_time
        total_training_time += epoch_training_time
        steps_in_epoch = len(train_loader)  # Total steps for this epoch (batches)
        total_steps += steps_in_epoch
        
        # Log the average training time per step for this epoch
        avg_time_per_step = epoch_training_time / steps_in_epoch

        # Log metrics to MLflow
        mlflow.log_metric("avg_time_per_step", avg_time_per_step, step=epoch)
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(epoch_acc)
        history['val_accuracy'].append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Early stopping
        if callbacks['best_loss'] is None or val_loss < callbacks['best_loss'] - callbacks['min_delta']:
            callbacks['best_loss'] = val_loss
            callbacks['counter'] = 0
            best_model = model
            print('val_loss improved, saving model')
            
        else:
            callbacks['counter'] += 1
            if callbacks['counter'] >= callbacks['patience']:
                callbacks['early_stop'] = True
                callbacks['counter'] = 0
                print("Early stopping")
                break

    example_input, example_output = example_data_loader()
    signature = infer_signature(example_input.cpu().numpy(), example_output.cpu().detach().numpy())
    
    # Log the model to MLflow at the end of training
    torch.save(best_model.state_dict(), callbacks['checkpoint_path'])

    mlflow.pytorch.log_model(best_model, 
                        artifact_path = "model",
                        signature = signature,
                        registered_model_name = "Classification-row-images"
                        )

    # Log the total average training time per step for all epochs
    overall_avg_time_per_step = total_training_time / total_steps
    mlflow.log_metric("overall_avg_time_per_step", overall_avg_time_per_step)
    

    return history


if __name__ == "__main__":
    print("\n","\n","Classifier training script", "\n")

    # Setup
    set_seed()
    setup_matplotlib()
 
    # Create the graphs folder if it doesn't exist
    graphs_folder = 'graphs'
    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)
    
    # Constants
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 20
    EPOCHS = 5
    DATAPATH = 'datasets/classification_balanced'
    SAVEPATH = 'models'

    class_names = sorted([d.name for d in os.scandir(DATAPATH) if d.is_dir()])
    
    # Create data generators
    train_loader, val_loader = create_data_generators(DATAPATH, IMG_SIZE, BATCH_SIZE)

    # Create and compile the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(IMG_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_model = model

    mlflow.set_experiment('Classification-row-images')
    run_dt = f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    print("Training...", "\n")

    with mlflow.start_run(run_name=f"{run_dt}") as run:

        run_name = f"{run_dt}-0.001"  

        # Define early stopping callback
        early_stopping, checkpoint_path = get_callbacks(SAVEPATH)
        early_stopping['checkpoint_path'] = checkpoint_path
        
        with mlflow.start_run(run_name=f"{run_name}", nested = True) as run:

            # Train the model
            history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, early_stopping, device)

        history_df = pd.DataFrame(history).fillna(np.nan)
        plot_history(history_df, save=False)

        # Recompile the model with a lower learning rate
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Define early stopping callback for fine-tuning
        early_stopping['min_delta'] = 0.0001
        early_stopping['counter'] = 0
        early_stopping['early_stop'] = False

        run_name = f"{run_dt}-0.0001"

        with mlflow.start_run(run_name=f"{run_name}", nested = True) as run:

            print("\n","Continuing with lower learning rate...", "\n")
            history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, early_stopping, device)

        history_df = pd.concat([history_df, pd.DataFrame(history).fillna(np.nan)], ignore_index=True)
        plot_history(history_df, save=False)

        print("\n", "NOW FINE-TUNING THE MODEL", "\n")

        # Unfreeze some layers and fine-tune the model
        for param in model.parameters():
                param.requires_grad = True

        # Recompile the model with a lower learning rate
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        run_name = f"{run_dt}-finetuning"
        with mlflow.start_run(run_name=f"{run_name}", nested = True) as run:

            # Continue training the model
            history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, early_stopping, device)
            
    print('Training completed. Model saved to:', SAVEPATH, '\n')    
    history_df = pd.concat([history_df, pd.DataFrame(history).fillna(np.nan)], ignore_index=True)

    plot_history(history_df, save=True)
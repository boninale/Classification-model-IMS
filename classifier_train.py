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

def create_data_generators(datapath, val_path = None, batch_size=25, val_split=0.2, split = True):
    """Create training and validation data generators."""
    transform = transforms.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(root=datapath, transform=transform)
    
    if split : 
        # Calculate the number of samples for training and validation
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        # Split the dataset
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    elif val_path:  
        train_dataset = full_dataset
        val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
        print(f'Using validation set from {val_path}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def create_model(model_path=None):
    """Create and compile the model."""
    if model_path:
        # Load the model architecture
        model = models.efficientnet_b0(weights=None)
        
        # Modify the classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3),
            nn.Softmax(dim=1)
        )
        
        # Load the pretrained weights
        model.load_state_dict(torch.load(model_path))
    else:
        # Create a new model with or without pretrained weights
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Modify the classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3),
            nn.Softmax(dim=1)
        )
    
    return model

def get_callbacks(patience):
    """Create callbacks for training."""
    early_stopping = {
        'patience': patience,
        'min_delta': 0.001,
        'counter': 0,
        'best_loss': None,
        'early_stop': False
    }

    return early_stopping

def example_data_loader():
    # Assuming typical input is a batch of images from the data loader
    train_loader, _ = create_data_generators(DATAPATH, 1)
    example_input, _ = next(iter(train_loader))
    example_input = example_input.to(device)
    example_output = model(example_input)

    return example_input, example_output

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, callbacks, device):
    """Train the model."""
    best_model = model.to(device)

    total_training_time = 0  # Track total training time for all epochs
    total_steps = 0  # Track total steps across all epochs

    # Log model parameters (e.g., hyperparameters)
    mlflow.log_param("learning_rate", optimizer.defaults['lr'])
    mlflow.log_param("batch_size", train_loader.batch_size)
    mlflow.log_param("epochs", EPOCHS)

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


        print(f'Epoch {epoch}/{epochs} : Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
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
                model = best_model # Load the best model found during training
                print("Early stopping")
                break


    # Log validation metrics as tags
    for i, recall in enumerate(recall_per_class):
        mlflow.set_tag(f"val_recall_{class_names[i]}", recall)

    # Log the total average training time per step for all epochs
    overall_avg_time_per_step = total_training_time / total_steps
    mlflow.log_metric("overall_avg_time_per_step", overall_avg_time_per_step)



def retrieve_metrics(metrics_to_retrieve) :
    """Retrieve metrics from the MLflow run."""
    client = mlflow.tracking.MlflowClient()
    run_info = mlflow.active_run().info
    run_id = run_info.run_id

    # Collect run parameters and metrics
    for metric in metrics_to_retrieve:
        metric_history = client.get_metric_history(run_id, metric)
        metrics_log[metric].extend(metric_history) #Append to previous history

    steps = max(len(records) for records in metrics_log.values())

    return metrics_log, steps 

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
    EPOCHS = 30
    PATIENCE = 10
    DATAPATH = 'datasets/ProspectFD/Minervois/CameraA/p0901_1433_A_balanced'
    SAVEPATH = 'models'
    VAL_PATH = 'datasets/test'
    model_path ='models/Run_2024-10-08_11-24-18.pth'

    print(f'Run parameters: \n - Batch size: {BATCH_SIZE} \n - Epochs: {EPOCHS} \n - Patience: {get_callbacks(SAVEPATH)["patience"]} \n - Data path: {DATAPATH} \n - Save path: {SAVEPATH} \n - Validation set: {VAL_PATH != None} \n - Pretrained Model: {model_path != None} \n')

    class_names = sorted([d.name for d in os.scandir(DATAPATH) if d.is_dir()])
    
    # Create data generators
    train_loader, val_loader = create_data_generators(DATAPATH, VAL_PATH, BATCH_SIZE ,split = False)

    # Create and compile the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "\n")
    model = create_model(model_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_model = model.to(device)

    mlflow.set_experiment('Classification-row-images-prospectFD')
    run_dt = f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    metrics_to_retrieve = [ "val_loss", 
                            "val_accuracy", 
                            "train_loss", 
                            "train_accuracy", 
                            "overall_avg_time_per_step",
                            "val_recall_missing_vine",
                            "val_recall_turn",
                            "val_recall_vine"
                            ]
    
    # Initialize a dictionary to store the metrics
    metrics_log = {metric: [] for metric in metrics_to_retrieve}

    steps = 0


    with mlflow.start_run(run_name=f"{run_dt}") as run:
        
        run_name = run_dt

        # Log model parameters (e.g., hyperparameters)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("patience", get_callbacks(PATIENCE)['patience'])
        # Set some tags for metadata
        mlflow.set_tag("model_type", "EfficientNetB0")
        #mlflow.set_tag("note", "First run using MLflow")

        run_name = f"{run_dt}-0.001"  

        # Define early stopping callback
        early_stopping = get_callbacks(PATIENCE)
    
        
        with mlflow.start_run(run_name=f"{run_name}", nested = True) as run:
            
            # Train the model
            train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, early_stopping, device)
            metrics_log, steps = retrieve_metrics(metrics_to_retrieve)
        
        # history_df = pd.DataFrame(history).fillna(np.nan)
        # plot_history(history_df, save=False)

        # Recompile the model with a lower learning rate
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Define early stopping callback for fine-tuning
        early_stopping['min_delta'] = 0.0001
        early_stopping['counter'] = 0
        early_stopping['early_stop'] = False

        run_name = f"{run_dt}-0.0001"

        with mlflow.start_run(run_name=f"{run_name}", nested = True) as run:

            print("\n","Continuing with lower learning rate...", "\n")
            # Train the model
            train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, early_stopping, device)
            metrics_log, steps = retrieve_metrics(metrics_to_retrieve)

        print("\n", "NOW FINE-TUNING THE MODEL", "\n")

        # Unfreeze some layers and fine-tune the model
        for param in model.parameters():
                param.requires_grad = True

        # Recompile the model with a lower learning rate
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        run_name = f"{run_dt}-finetuning"
        with mlflow.start_run(run_name=f"{run_name}", nested = True) as run:
            # Train the model
            train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, early_stopping, device)
            metrics_log, steps = retrieve_metrics(metrics_to_retrieve)

        # Save the model
        torch.save(model.state_dict(), os.path.join(SAVEPATH, f'{run_dt}.pth'))

        # Log the model to MLflow
        example_input, example_output = example_data_loader()
        signature = infer_signature(example_input.cpu().numpy(), example_output.cpu().detach().numpy())

        mlflow.pytorch.log_model(pytorch_model = best_model, 
                            artifact_path="model",
                            signature=signature,
                            registered_model_name="Classification-row-images-prospectFD"
                            )

        print('Training completed. Model saved to:', SAVEPATH, '\n')    

        # Log the metrics and parameters to the main run
        for step in range(steps):
            for metric in metrics_to_retrieve:
                if step < len(metrics_log[metric]):
                    mlflow.log_metric(metric, metrics_log[metric][step].value, step=step)

        # Find the index at which val_accuracy was the highest
        max_val_accuracy_index = max(range(len(metrics_log['val_accuracy'])), key=lambda i: metrics_log['val_accuracy'][i].value)

        # Log the metrics for the index at which val_accuracy was the highest
        for metric in metrics_to_retrieve:
            if max_val_accuracy_index < len(metrics_log[metric]):
                mlflow.log_metric(metric, metrics_log[metric][max_val_accuracy_index].value, step=steps+1)
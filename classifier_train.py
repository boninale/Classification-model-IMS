import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import time
from datetime import datetime
from sklearn.metrics import recall_score

from architectures import EffNetB0, CombinedHeadModel

#region Functions
def set_seed(seed=31415):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_data_generators(datapath, val_path = None, batch_size=25, val_split=0.2):

    """Create training and validation data generators."""
    transform = transforms.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(root=datapath, transform=transform)
    
    if val_path==None: 

        # Calculate the number of samples for training and validation
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        # Split the dataset
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    else:  
        train_dataset = full_dataset
        val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
        print(f'Using validation set from {val_path}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def get_callbacks(patience = 10):
    """Create callbacks for training."""
    early_stopping = {
        'patience': patience,

        'counter': 0,
        'best_loss': None,
        'early_stop': False
    }

    return early_stopping

def example_data_loader():
    # Assuming typical input is a batch of images from the data loader
    train_loader, _ = create_data_generators(DATAPATH, val_path=None, batch_size=BATCH_SIZE, val_split=0)
    example_input = next(iter(train_loader))[0]
    example_input = example_input.to(device)
    example_output = model(example_input)

    return example_input, example_output

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, callbacks, device):
    """Train the model."""
    best_model_wts = model.state_dict()
    callbacks['best_loss'] = min_val_loss

    model = model.to(device)

    total_training_time = 0  # Track total training time for all epochs
    total_steps = 0  # Track total steps across all epochs
    callbacks['counter'] = 0

    # Log model parameters (e.g., hyperparameters)
    mlflow.log_param("batch_size", train_loader.batch_size)
    mlflow.log_param("epochs", EPOCHS)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):

        start_time = time.time()  # Start timing the epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(): # Mixed precision training
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward() 
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        end_time = time.time()  # End timing the epoch
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        scheduler1.step()
        scheduler2.step(epoch_loss)
        
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
            mlflow.log_metric(f"val_recall_{class_names[i]}", recall, step= STEPS + epoch)
         
        epoch_training_time = end_time - start_time
        total_training_time += epoch_training_time
        steps_in_epoch = len(train_loader)  # Total steps for this epoch (batches)
        total_steps += steps_in_epoch
        
        # Log the average training time per step for this epoch
        avg_time_per_step = epoch_training_time / steps_in_epoch

        # Log metrics to MLflow
        mlflow.log_metric("avg_time_per_step", avg_time_per_step, step=STEPS + epoch)
        mlflow.log_metric("train_loss", epoch_loss, step=STEPS + epoch)
        mlflow.log_metric("train_accuracy", epoch_acc, step=STEPS + epoch)
        mlflow.log_metric("val_loss", val_loss, step=STEPS + epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=STEPS + epoch)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=STEPS + epoch)

        print(f'Epoch {epoch}/{epochs} : Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Early stopping
        if callbacks['best_loss'] is None or val_loss < callbacks['best_loss']:
            callbacks['best_loss'] = val_loss
            callbacks['counter'] = 0
            best_model_wts = model.state_dict()
            print('val_loss improved, saving model')
            
        else:
            callbacks['counter'] += 1
            if callbacks['counter'] >= callbacks['patience']:
                callbacks['early_stop'] = True
                callbacks['counter'] = 0
                print("Early stopping")
                break
                
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Log validation metrics as tags
    for i, recall in enumerate(recall_per_class):
        mlflow.set_tag(f"val_recall_{class_names[i]}", recall)

    # Log the total average training time per step for all epochs
    overall_avg_time_per_step = total_training_time / total_steps
    mlflow.log_metric("overall_avg_time_per_step", overall_avg_time_per_step)


def get_metrics() :
    """Retrieve metrics from the MLflow run."""
    client = mlflow.tracking.MlflowClient()
    run_info = mlflow.active_run().info
    run_id = run_info.run_id

    # Collect run parameters and metrics
    steps = len(client.get_metric_history(run_id, "val_loss"))
    min_val_loss = min([m.value for m in client.get_metric_history(run_id, "val_loss")])
    max_val_acc = max([m.value for m in client.get_metric_history(run_id, "val_accuracy")])

    return steps, min_val_loss, max_val_acc

# endregion

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
    
    # region Initialization

    # Setup
    #set_seed()

    # Constants
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 20
    EPOCHS = 30
    PATIENCE = 2
    DATAPATH = 'datasets/ProspectFD/Minervois/CameraA/p0901_1433_A_balanced'
    SAVEPATH = 'models'
    VAL_PATH = None
    model_path = None
    new_model_name = 'dev'
    experiment = 'Classification-row-images-dev'
    model_type = 'HeadV2'
    class_names = sorted([d.name for d in os.scandir(DATAPATH) if d.is_dir()])
    num_classes = len(class_names)
    
    # Create data generators
    train_loader, val_loader = create_data_generators(DATAPATH, VAL_PATH, BATCH_SIZE)

    # Create and compile the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "\n")
    model = EffNetB0(num_classes, model_path)  #CombinedHeadModel(embedding_dim = 3, num_classes = 3, model_path = model_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler1 = ExponentialLR(optimizer, gamma=0.95)
    scheduler2 = ReduceLROnPlateau(optimizer, mode='min', factor=0.60, patience=PATIENCE//2)
    best_model_wts = model.state_dict()

    print(f'Run parameters: \n - Batch size: {BATCH_SIZE} \n - Epochs: {EPOCHS} \n - Patience: {get_callbacks(PATIENCE)["patience"]} \n - Data path: {DATAPATH} \n - Save path: {SAVEPATH} \n - Validation set: {VAL_PATH != None} \n - Pretrained Model: {model_path != None} \n')

    mlflow.set_experiment(experiment)
    run_dt = f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    metrics_to_retrieve = [ "val_loss", 
                            "val_accuracy",
                            "learning_rate", 
                            "train_loss", 
                            "train_accuracy", 
                            "overall_avg_time_per_step",
                            "val_recall_missing_vine",
                            "val_recall_turn",
                            "val_recall_vine"
                            ]
    
    # Initialize a dictionary to store the metrics
    metrics_log = {metric: [] for metric in metrics_to_retrieve}
    # endregion
    #region Training

    STEPS = 0
    min_val_loss = None
    max_val_acc = None

    with mlflow.start_run(run_name=f"{run_dt}") as run:
        
        run_name = run_dt

        # Log model parameters (e.g., hyperparameters)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("patience", get_callbacks(PATIENCE)['patience'])
        mlflow.log_param("dataset", DATAPATH)

        # Set some tags for metadata
        mlflow.set_tag("model_type", model_type)
        #mlflow.set_tag("note", "First run using MLflow")

        # Define early stopping callback
        early_stopping = get_callbacks(PATIENCE)
    
        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, early_stopping, device)
        STEPS , min_val_loss , max_val_acc = get_metrics()

        print("\n", "NOW FINE-TUNING THE MODEL", "\n")

        # Unfreeze some layers and fine-tune the model
        for param in model.parameters():
                param.requires_grad = True

        # Recompile the model with a lower learning rate
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler1 = ExponentialLR(optimizer, gamma=0.99)
        scheduler2 = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=PATIENCE//2)

        #run_name = f"{run_dt}-finetuning"

        #with mlflow.start_run(run_name=f"{run_name}", nested = True) as run:
            # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, early_stopping, device)
        STEPS , min_val_loss, max_val_acc = get_metrics()

        mlflow.log_metric("val_accuracy", max_val_acc, step = STEPS)
        
        # Save the model
        torch.save(model.state_dict(), os.path.join(SAVEPATH, f'{run_dt}.pth'))
        # Log the model to MLflow
        example_input, example_output = example_data_loader()
        signature = infer_signature(example_input.cpu().numpy(), example_output.cpu().detach().numpy())

        mlflow.pytorch.log_model(pytorch_model = model, 
                            artifact_path="model",
                            signature=signature,
                            registered_model_name=new_model_name
                            )

        print('Training completed. Model saved to:', SAVEPATH, '\n')    

       
        #endregion
    mlflow.end_run()


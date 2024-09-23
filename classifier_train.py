print("\n","\n","Classifier training script","\n")

## Setup


import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import datetime


# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')


# Setup plotting

plt.style.use('ggplot')

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')
plt.rc('image', cmap='magma')
rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False


# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 20
EPOCHS = 30

#Add original dataset path
DATAPATH = 'datasets/classification_balanced' 

#Path to save final model weights
SAVEPATH = 'models'

## Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATAPATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATAPATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

## Model training

# Load the ResNet50 model without the top layer
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
# Add custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of ResNet50
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

# device_name = tf.test.gpu_device_name()
# if not device_name:
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch='10, 15')


#Define early stopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', # what to monitor
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=False,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(SAVEPATH + '/EffNetB0_classifier.keras', 
                    monitor="val_loss", mode="min", 
                    save_best_only=True, verbose=2)

print("Training...", "\n")
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs= EPOCHS,
    callbacks=[early_stopping, checkpoint, tensorboard_callback],
    verbose = 1
)


# Recompile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#Define early stopping callback
early_stopping = callbacks.EarlyStopping(
    min_delta=0.0001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)

print("Continuing with lower learning rate ...", "\n")
# Continue training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint],
    verbose = 2
)


print('Training completed. Model saved to:', SAVEPATH,'\n')

# Create the graphs folder if it doesn't exist
graphs_folder = 'graphs'
if not os.path.exists(graphs_folder):
    os.makedirs(graphs_folder)

# history is the History object returned by model.fit()
history_df = pd.DataFrame(history.history)


# Plot training and validation loss
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(graphs_folder, 'training_validation_loss.png'))  # Save the plot
plt.show()

print("\n", "Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()), "\n")
print("Corresponding Validation Accuracy: {:0.4f}".format(history_df['val_accuracy'][history_df['val_loss'].idxmin()]), "\n");

print("\n", "NOW FINE TUNING THE MODEL", "\n")
# Unfreeze some layers and fine-tune the model
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#Define early stopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', # what to monitor
    min_delta=0.0001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=False,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(SAVEPATH + '/EffNetB0_classifier_finetuned.keras', 
                    monitor="val_loss", mode="min", 
                    save_best_only=True, verbose=2)

# Continue training the model
train = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint],
    verbose = 2
)

print('Training completed. Model saved to:', SAVEPATH,'\n')


# history is the History object returned by model.fit()
history_df = pd.DataFrame(history.history)


# Plot training and validation loss
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(graphs_folder, 'training_validation_loss_finetuned.png'))  # Save the plot
plt.show()

print("\n", "Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()), "\n")
print("Corresponding Validation Accuracy: {:0.4f}".format(history_df['val_accuracy'][history_df['val_loss'].idxmin()]), "\n");

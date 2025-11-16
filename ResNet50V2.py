import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout,
    RandomFlip, RandomRotation, RandomZoom, Rescaling, RandomContrast,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


num_classes = 6
IMG_SIZE = 128  # Resize all images to 128x128 pixels
labels = ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']

X, y = [], []  # X: image data, y: labels

dataset_path = r'NEU-DET/train/images/'


# Loop through each defect type folder
for idx, label in enumerate(labels):
    folder = os.path.join(dataset_path, label)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if img is None:
            continue  # Skip if image cannot be read
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
        X.append(img)
        y.append(idx)  # Assign numeric label


# Convert lists to NumPy arrays and normalize pixel values
# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

# Convert (N, H, W) grayscale to (N, H, W, 3) "fake" RGB
X = np.stack([X, X, X], axis=-1).astype("float32")
# print("Converted X shape for RGB:", X.shape)

# y = to_categorical(np.array(y), num_classes=len(labels))  # One-hot encoding for labels
y_int = np.array(y)

# Split dataset into training and testing sets
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X, y_int, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_int  # <-- This is the crucial fix
)

# --- Now, one-hot encode *both* the train and test labels
y_train = to_categorical(y_train_int, num_classes=num_classes)
y_test = to_categorical(y_test_int, num_classes=num_classes)

print(X)
print(y_train)
print(X.shape)
print(y_train.shape)

# -----------------------------
# 2. Build CNN Model (TRANSFER LEARNING)
# -----------------------------

# Define the input shape
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# --- Augmentation Layers ---
# We apply augmentation *before* the model
x = RandomFlip("horizontal")(inputs)
x = RandomRotation(0.10)(x)
x = RandomZoom(0.10)(x)
x = RandomContrast(0.1)(x)

# --- Pre-processing ---
# MobileNetV2 expects inputs in the range -1 to 1, not 0 to 1
# We use its dedicated preprocessing function.
# NOTE: We do *not* use your Rescaling(1./255) layer anymore.
x = tf.keras.applications.resnet_v2.preprocess_input(x)

base_model = ResNet50V2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# FREEZE the base model
base_model.trainable = False

# Pass our augmented data *through* the frozen base
x = base_model(x, training=False) # training=False ensures BN layers stay frozen

# --- Classifier Head (Trainable) ---
# This is *our* part of the model
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Regularization
x = Dense(64, activation='relu')(x) # A small dense layer
x = Dense(num_classes, activation='softmax')(x) # Output layer

# --- Final Model ---
model = Model(inputs, x)

# --- Compile ---
# We can start with a slightly higher LR for transfer learning
model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    ModelCheckpoint('best_steel_defect_transfer.keras', save_best_only=True, monitor='val_loss')
]

# -----------------------------
# 3. Train the Model
# -----------------------------
print("Starting model training...")
history = model.fit(
    X_train, y_train, 
    epochs=40,  # Train for *more* epochs; EarlyStopping will find the best one
    batch_size=32,
    validation_split=0.1,  # This is great!
    callbacks=callbacks      # <-- Pass the callbacks here
)


# Create a figure with 2 subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

# -----------------------------
# Plot Accuracy
# -----------------------------
ax1.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# -----------------------------
# Plot Loss
# -----------------------------
ax2.plot(history.history['loss'], label='Train Loss', marker='o')
ax2.plot(history.history['val_loss'], label='Validation Loss', marker='o')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# Show the combined figure
plt.tight_layout()
plt.show()


# Evaluate model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save trained model for later use
model.save('steel_defect_resnet50v2.keras')
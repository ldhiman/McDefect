import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(np.array(y), num_classes=len(labels))  # One-hot encoding for labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X)
print(y)
print(X.shape)
print(y.shape)

# -----------------------------
# 2. Build CNN Model
# -----------------------------
model = Sequential([
    # First convolutional layer
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE,1)),
    MaxPooling2D(2,2),  # Max pooling layer to reduce spatial dimensions

    # Second convolutional layer
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # Flatten 2D feature maps to 1D
    Flatten(),

    # Fully connected layer
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout to reduce overfitting

    # Output layer with softmax for multi-class classification
    Dense(len(labels), activation='softmax')
])

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# -----------------------------
# 3. Train the Model
# -----------------------------
history = model.fit(
    X_train, y_train, 
    epochs=20,  # Number of epochs (can reduce for testing)
    batch_size=32,
    validation_split=0.1  # Use 10% of training data for validation
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
model.save('steel_defect_cnn_v1.keras')
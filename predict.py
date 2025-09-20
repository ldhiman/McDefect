import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


# -----------------------------
# 4. GUI for Real-Time Prediction
# -----------------------------
def predict_image():
    """
    Open a file dialog to select an image,
    preprocess it, and predict the defect type.
    """
    file_path = filedialog.askopenfilename()  # Open file chooser
    if not file_path:
        return  # Exit if no file selected

    # Read image in grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
    img_array = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize

    # Predict defect type
    pred = model.predict(img_array)
    label_pred = labels[np.argmax(pred)]  # Convert one-hot to label

    # Display result in GUI
    result_label.config(text=f"Predicted Defect: {label_pred}")

# Create Tkinter window
root = Tk()
root.title("Steel Surface Defect Detection")
root.geometry("400x200")

# Button to select image
btn = Button(root, text="Select Image", command=predict_image, font=("Arial", 14))
btn.pack(pady=20)

# Label to display prediction
result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

# Run GUI
root.mainloop()
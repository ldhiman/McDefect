import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# ================================
# CONFIG
# ================================
IMG_SIZE = 128
labels = ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
num_classes = len(labels)

# <<< DEFINE YOUR MODELS HERE >>>
MODEL_PATHS = [
    "steel_defect_cnn_v1.keras",
    "steel_defect_mobileVnet_v1.keras",
    "steel_defect_efficientnet_b0.keras",
    "steel_defect_DenseNet121.keras",
    "steel_defect_ResNet50V2.keras",
]

# Test dataset folder (one subfolder per class)
TEST_DATASET_PATH = r"NEU-DET/validation/images/"

# Output folder
OUT_DIR = "cm_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ================================
# LOAD TEST IMAGES (GRAYSCALE)
# ================================
X_test, y_test = [], []

for idx, label in enumerate(labels):
    folder = os.path.join(TEST_DATASET_PATH, label)
    if not os.path.isdir(folder):
        raise RuntimeError(f"Missing folder: {folder}")

    for file in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X_test.append(img)
        y_test.append(idx)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32")
y_test = np.array(y_test)

print("Loaded Test Set:", X_test.shape, y_test.shape)

# ================================
# ADAPT IMAGES FOR EACH MODEL
# ================================
def adapt_images_for_model(X_gray, model):
    """
    Convert grayscale images (N,H,W,1) -> model input size / channels.
    Handles:
        - resizing
        - 1->3 channels (for MobileNet)
        - 3->1 (if needed)
        - trimming/expanding channels
    """
    input_shape = model.input_shape
    
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    try:
        _, h, w, c = input_shape
    except:
        h, w, c = IMG_SIZE, IMG_SIZE, 3   # fallback

    # Replace None
    h = IMG_SIZE if h is None else h
    w = IMG_SIZE if w is None else w
    c = 3 if c is None else c

    h, w, c = int(h), int(w), int(c)

    # Resize if needed
    if (h != X_gray.shape[1]) or (w != X_gray.shape[2]):
        X_resized = np.zeros((X_gray.shape[0], h, w, 1), dtype=X_gray.dtype)
        for i in range(X_gray.shape[0]):
            X_resized[i,:,:,0] = cv2.resize(X_gray[i].squeeze(), (w, h))
    else:
        X_resized = X_gray.copy()

    # Convert channels
    if c == 3 and X_resized.shape[-1] == 1:
        X_out = np.repeat(X_resized, 3, axis=-1)
    elif c == 1 and X_resized.shape[-1] == 3:
        X_out = np.mean(X_resized, axis=-1, keepdims=True)
    elif c == X_resized.shape[-1]:
        X_out = X_resized
    else:
        # General case (rare)
        X_out = np.repeat(X_resized, c, axis=-1)[:, :, :, :c]

    return X_out

# ================================
# CONFUSION MATRIX PLOTTING
# ================================
def save_confusion_matrix(cm, classes, outpath, normalize=False):
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)

    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(os.path.basename(outpath))
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i,j], fmt),
            horizontalalignment="center",
            color="white" if cm[i,j] > thresh else "black"
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved:", outpath)

# ================================
# EVALUATE MULTIPLE MODELS
# ================================
for model_path in MODEL_PATHS:
    print("\n==============================")
    print(" Evaluating:", model_path)
    print("==============================")

    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        continue

    model = load_model(model_path)
    print("Model loaded.")

    # Adapt test images
    X_input = adapt_images_for_model(X_test, model)
    print("Adapted input shape:", X_input.shape)

    # Predict
    probs = model.predict(X_input, batch_size=32)
    y_pred = np.argmax(probs, axis=1)

    print("Prediction distribution:", np.bincount(y_pred, minlength=num_classes))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    base = os.path.splitext(os.path.basename(model_path))[0]
    raw_out = f"cm_outputs/cm_{base}.png"
    norm_out = f"cm_outputs/cm_{base}_norm.png"

    save_confusion_matrix(cm, labels, raw_out, normalize=False)
    save_confusion_matrix(cm, labels, norm_out, normalize=True)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=labels))

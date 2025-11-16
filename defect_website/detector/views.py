from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import numpy as np
import cv2
import os
import base64

# --- Constants ---
IMG_SIZE = 128
LABELS = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# --- 1. Model Configuration (FIXED) ---
MODELS_CONFIG = {
    'efficientnet_b0': {
        'name': 'EfficientNetB0',
        'file': 'steel_defect_efficientnet_b0.keras',
        'preprocess': 'efficientnet' # Expects 3-channel, 0-255
    },
    'resnet50v2': {
        'name': 'ResNet50V2',
        'file': 'steel_defect_resnet50v2.keras',
        'preprocess': 'mobilenet' # Expects 3-channel, -1 to 1
    }, 
    'densenet121': {
        'name': 'DenseNet121',
        'file': 'steel_defect_DenseNet121.keras',
        'preprocess': 'densenet' # Expects 3-channel, [0,1] + norm
    },
    'mobilenet_v2': {
        'name': 'MobileNetV2',
        'file': 'steel_defect_mobileVnet_v2.keras',
        'preprocess': 'mobilenet' # Expects 3-channel, -1 to 1
    },
    'custom_cnn': {
        'name': 'Normal CNN',
        'file': 'steel_defect_cnn_v1.keras',
        'preprocess': 'cnn' # Expects 1-channel, 0 to 1
    }
}

# --- 2. Load all models (do this ONCE at startup) ---
LOADED_MODELS = {}
for key, config in MODELS_CONFIG.items():
    model_path = os.path.join(settings.BASE_DIR, 'detector', config['file'])
    try:
        model = load_model(model_path)
        LOADED_MODELS[key] = model
        print(f"Successfully loaded model '{key}' from {model_path}")
    except Exception as e:
        print(f"Error loading model '{key}': {e}")
        LOADED_MODELS[key] = None

# --- 3. Preprocessing Functions ---

def _preprocess_mobilenet(image_bytes):
    """ Prepares image for MobileNetV2 & ResNet50V2 (3-channel, -1 to 1) """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return None, "Unable to decode image."
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)
    img_batch = np.expand_dims(img_rgb, axis=0).astype("float32")
    return mobilenet_preprocess(img_batch), None

def _preprocess_custom(image_bytes):
    """ Prepares image for the custom CNN (1-channel, 0 to 1) """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return None, "Unable to decode image."
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_batch = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
    return img_batch, None

def _preprocess_efficientnet(image_bytes):
    """ Prepares image for EfficientNetB0 (3-channel, 0-255) """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return None, "Unable to decode image."
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)
    img_batch = np.expand_dims(img_rgb, axis=0).astype("float32")
    return img_batch, None

def _preprocess_densenet(image_bytes):
    """ Prepares image for DenseNet121 (3-channel, [0,1] + norm) """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return None, "Unable to decode image."
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)
    img_batch = np.expand_dims(img_rgb, axis=0).astype("float32")
    return densenet_preprocess(img_batch), None


def preprocess_image(image_bytes, preprocess_type):
    """ Main dispatcher function for preprocessing """
    if preprocess_type == 'mobilenet':
        return _preprocess_mobilenet(image_bytes)
    elif preprocess_type == 'cnn':
        return _preprocess_custom(image_bytes)
    elif preprocess_type == 'efficientnet':
        return _preprocess_efficientnet(image_bytes)
    elif preprocess_type == 'densenet':
        return _preprocess_densenet(image_bytes)
    else:
        return None, f"Unknown preprocessing type: {preprocess_type}"

# --- 4. The main view for handling requests ---
def index(request):
    context = {
        'models': MODELS_CONFIG
    }

    if request.method == 'POST' and request.FILES.get('imagefile'):
        model_key = request.POST.get('model_choice')
        uploaded_file = request.FILES['imagefile']
        file_bytes = uploaded_file.read()
        
        # --- Prepare context for the results page ---
        context.update({
            'selected_model_key': model_key,
            'file_name': uploaded_file.name,
            'uploaded_image': base64.b64encode(file_bytes).decode('utf-8'),
            'results': [] # We will always send a results list
        })
        
        # --- Determine which models to run ---
        models_to_run = []
        if model_key == 'all':
            # Add all loaded models
            for key, model in LOADED_MODELS.items():
                if model: # Only add if it loaded correctly
                    models_to_run.append((key, model))
        else:
            # Add just the selected model
            if model_key in LOADED_MODELS and LOADED_MODELS[model_key]:
                models_to_run.append((model_key, LOADED_MODELS[model_key]))
            else:
                context['error'] = f"Model '{model_key}' is not available."
                return render(request, 'detector/index.html', context)

        # --- Loop and predict for each model ---
        for key, model in models_to_run:
            config = MODELS_CONFIG[key]
            preprocess_type = config['preprocess']
            
            result_data = {'name': config['name']}
            
            # Preprocess the image
            processed_image, error = preprocess_image(file_bytes, preprocess_type)
            
            if error:
                result_data['error'] = f"Preprocessing failed: {error}"
            else:
                # --- DEBUGGING ---
                print(f"\n--- Running: {config['name']} ---")
                print(f"Preprocessing: {preprocess_type}")
                print(f"Shape: {processed_image.shape}, Min: {processed_image.min():.2f}, Max: {processed_image.max():.2f}")
                
                # --- Predict ---
                prediction = model.predict(processed_image)
                pred_idx = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction, axis=1)[0] * 100
                
                result_data['prediction'] = LABELS[pred_idx]
                result_data['confidence'] = f"{confidence:.2f}"
            
            context['results'].append(result_data)

    # For a GET request, or after a POST, render the page
    return render(request, 'detector/index.html', context)
# McDefect: Steel Surface Defect Classification Model Evaluation

This project evaluates and compares the performance of several pre-trained Deep Learning models for the task of classifying steel surface defects. It uses the North-Eastern University (NEU) surface defect dataset.

**Student Name:** Paramjeet, Paras, Priyanshu \
**Roll Number:** 2K22/ME/183, 2K22/ME/184, 2K22/ME/204 \
**Department:** Mechanical Engineering \
**Semester:** 7th \
**Guide Name:** Prof. Rooplal \
**College:** Delhi Technological University \
**Session:** 2025-26

!Sample Confusion Matrix

## Features

- **Multi-Model Evaluation**: Sequentially evaluates a list of specified Keras models.
- **Data Preprocessing**: Automatically loads, resizes, and adapts images (e.g., grayscale to RGB) to match the input requirements of each model.
- **Performance Metrics**: For each model, it:
  - Generates and saves a raw count confusion matrix image.
  - Generates and saves a normalized confusion matrix image.
  - Prints a detailed classification report (precision, recall, F1-score) to the console.

## Dataset

This project uses the **NEU Surface Defect Database**. The script is configured to read images from a specific folder structure.

- **Classes**: The models are trained to classify 6 types of defects:
  1.  `crazing`
  2.  `inclusion`
  3.  `patches`
  4.  `pitted_surface`
  5.  `rolled-in_scale`
  6.  `scratches`

You must download the dataset and organize the validation images into subfolders named after these classes.

## Models Evaluated

The script is configured to evaluate the following model architectures. You must provide the corresponding `.keras` model files.

- `steel_defect_cnn_v1.keras` (Custom CNN)
- `steel_defect_mobileVnet_v1.keras` (MobileNetV2)
- `steel_defect_efficientnet_b0.keras` (EfficientNetB0)
- `steel_defect_DenseNet121.keras` (DenseNet121)
- `steel_defect_ResNet50V2.keras` (ResNet50V2)

## Prerequisites

- Python 3.8+
- The required Python libraries can be installed from `requirements.txt`.

```
tensorflow
scikit-learn
matplotlib
numpy
opencv-python
```

You can install them all with pip:

```bash
pip install -r requirements.txt
```

## Setup & Usage

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Install dependencies:**

    ```bash
    pip install tensorflow scikit-learn matplotlib opencv-python numpy
    ```

3.  **Prepare the data:**

    - Download the NEU Surface Defect Database.
    - Create a directory structure as expected by the script: `NEU-DET/validation/images/`.
    - Inside the `images` folder, create a subfolder for each defect class (e.g., `crazing`, `inclusion`, etc.) and place the corresponding images inside.

4.  **Place the models:**

    - Place your pre-trained `.keras` model files in the root directory of the project. Ensure their names match the ones listed in `MODEL_PATHS` in the script.

5.  **Run the evaluation:**
    Execute the script from your terminal.
    ```bash
    python confusionMatrix.py
    ```

## Output

- **Console Output**: For each model, a classification report is printed, showing precision, recall, and F1-score for each class.
- **Image Files**: The script generates two confusion matrix images (`.png`) for each model and saves them in the `cm_outputs/` directory:
  - `cm_<model_name>.png`: Confusion matrix with raw prediction counts.
  - `cm_<model_name>_norm.png`: Confusion matrix with values normalized by the number of true instances per class (showing percentages).

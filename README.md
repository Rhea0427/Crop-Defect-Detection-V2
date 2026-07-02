# Crop Defect Detection

## Project Overview
This repository contains a complete image classification pipeline for sugarcane leaf disease detection. It includes dataset preprocessing, transfer learning with EfficientNetB0, training, model evaluation, prediction, Grad-CAM explainability, and a Streamlit demo app.

## What’s Included
- `main.py`: end-to-end pipeline to prepare data, train or load the model, evaluate performance, make a sample prediction, and save Grad-CAM output.
- `app.py`: Streamlit application for interactive image upload and disease prediction.
- `src/`: reusable modules for preprocessing, augmentation, model construction, training, evaluation, prediction, and Grad-CAM.
- `dataset/`: class-labeled sugarcane leaf images.
- `graphs/`: generated visualization output, including EDA charts and model training graphs.
- `models/`: saved model artifacts.
- `weights/`: saved model weights.
- `reports/`: saved metrics and classification reports.
- `predictions/`: prediction and Grad-CAM output images.

## Dataset Description
The dataset contains images of sugarcane leaves in five classes:
- Healthy
- Mosaic
- RedRot
- Rust
- Yellow

## Environment Setup
Create and activate a local Python virtual environment before installing dependencies:

```bash
cd Crop-Defect-Detection-V1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Note: On macOS, use the virtual environment to avoid system package conflicts with TensorFlow and OpenCV.

## How to Run
Run the main pipeline:

```bash
source .venv/bin/activate
python main.py
```

Run the Streamlit app:

```bash
source .venv/bin/activate
streamlit run app.py
```

## Output Files
After running the pipeline, the repository produces:
- `graphs/training_accuracy.png`
- `graphs/training_loss.png`
- `graphs/confusion_matrix.png`
- `reports/classification_report.txt`
- `predictions/gradcam_prediction.png`
- `models/best_model.keras`
- `weights/best_weights.keras`

## Notes
- Model training currently uses `EfficientNetB0` as a frozen base and trains a custom top layer.
- The project saves both the full model and separate weights.
- `main.py` will load an existing model if `models/best_model.keras` exists.

## Project Documentation
See `PROJECT_DOCUMENTATION.md` for a full project summary, architecture details, setup instructions, and next steps.

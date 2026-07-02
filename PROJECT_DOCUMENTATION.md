# Crop Defect Detection Project Documentation

## Project Summary
This project builds and evaluates a transfer learning-based deep learning classifier for sugarcane leaf disease detection. The pipeline uses TensorFlow and Keras to preprocess images, train a model with EfficientNetB0, and generate evaluation reports and explainability visualizations.

## Goals
- Prepare the sugarcane leaf dataset for training, validation, and testing.
- Build a robust transfer learning model using `EfficientNetB0`.
- Train the model with data augmentation and callbacks.
- Evaluate performance with a classification report and confusion matrix.
- Generate Grad-CAM heatmaps for model explainability.
- Provide a Streamlit app for interactive prediction.

## Architecture
- `src/preprocess.py`: loads image paths, splits data, decodes and normalizes images, and builds TensorFlow datasets.
- `src/augmentation.py`: applies image augmentations such as flipping, brightness, contrast, saturation, and hue.
- `src/model.py`: constructs the model architecture using EfficientNetB0 and custom dense layers.
- `src/train.py`: compiles the model, trains it, and saves training plots.
- `src/evaluate.py`: evaluates the model on the test set, writes a classification report, and saves a confusion matrix.
- `src/predict.py`: loads models, preprocesses input images, and performs single-image predictions.
- `src/grad_cam.py`: generates and saves Grad-CAM heatmaps.
- `main.py`: orchestrates the full pipeline from dataset preparation through evaluation and explanation.
- `app.py`: Streamlit interface for uploading images and viewing predictions.

## Dataset
The images are organized in `dataset/` by class. The pipeline automatically discovers classes and splits the data into:
- 70% training
- 15% validation
- 15% test

Supported image formats include `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, and `.webp`.

## Output Artifacts
- `graphs/`: contains both original EDA charts and model output graphs
- `reports/`: classification reports and metrics
- `models/`: saved Keras model files
- `weights/`: saved model weights
- `predictions/`: sample prediction images and Grad-CAM heatmaps



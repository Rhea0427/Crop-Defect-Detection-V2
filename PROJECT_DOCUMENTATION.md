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

## Setup and Installation
1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Upgrade pip and install requirements:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. On macOS, use the local `.venv` to avoid conflicts with system packages.

## Running the Pipeline
Execute the complete training and evaluation workflow:

```bash
source .venv/bin/activate
python main.py
```

The pipeline will:
- prepare the datasets
- train the model if none exists
- evaluate the model
- save classification reports and confusion matrices
- generate a sample prediction and Grad-CAM heatmap

## Running the Streamlit App
Start the interactive app:

```bash
source .venv/bin/activate
streamlit run app.py
```

Then open the provided local URL in your browser.

## Output Artifacts
- `graphs/`: contains both original EDA charts and model output graphs
- `reports/`: classification reports and metrics
- `models/`: saved Keras model files
- `weights/`: saved model weights
- `predictions/`: sample prediction images and Grad-CAM heatmaps

## Notes and Recommendations
- `main.py` currently prefers an existing saved model if available.
- The Grad-CAM visualizations help validate which regions influenced the model's predictions.
- The model currently uses basic augmentation and may benefit from further tuning and class balancing.

## Future Improvements
- add cross-validation or stratified splitting
- support more advanced augmentations and preprocessing
- implement a full training/serving pipeline for the Streamlit app
- improve model architecture with fine-tuning of EfficientNetB0
- add a requirements file with pinned package versions for reproducibility

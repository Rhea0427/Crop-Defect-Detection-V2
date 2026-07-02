# Crop-Defect-Detection-V1

## Project Overview
This project performs exploratory data analysis on a sugarcane leaf disease image dataset. It scans the dataset folders, counts images per class, computes descriptive statistics, generates visualizations, and writes a summary report. No deep learning model is trained.

## Dataset Description
The dataset contains images belonging to five classes:
- Healthy
- Mosaic
- RedRot
- Rust
- Yellow

## Folder Structure
- dataset/: contains the raw image folders
- graphs/: stores generated PNG visualizations
- notebooks/: place for exploratory notebooks
- reports/: stores reports and CSV outputs
- src/: reusable Python modules

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- scikit-learn
- Pillow

## Python Libraries
- pandas
- numpy
- matplotlib
- seaborn
- opencv-python
- scikit-learn
- Pillow
- tensorflow
- streamlit
- tensorflow-addons

## Screenshots Placeholder
Add screenshots of the generated charts here.

## How to Run
```bash
python main.py
```

## Expected Output
The script will generate:
- PNG charts in the graphs folder
- analysis_report.txt in the reports folder
- class_weights.csv in the reports folder

## Future Improvements
- Add image preprocessing and augmentation pipelines
- Integrate a classifier for disease prediction
- Add automated notebook templates for experimentation

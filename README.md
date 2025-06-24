# CNN-Cancer-Detection-Kaggle-Mini-Project

This project tackles the Kaggle challenge of detecting metastatic cancer in histopathologic image patches. The goal was to build a binary image classifier to determine the presence (1) or absence (0) of cancer in 96×96 .tif images.

## Project Highlights

- **Data Size**: 220,025 image-label pairs
- **Input**: RGB image patches (96×96), labeled 0 or 1
- **Models**:
  - Custom CNN trained on 5,000 balanced samples
  - Transfer learning with MobileNetV2 using pre-trained ImageNet weights

## Approach

1. **EDA**: 
   - Visualized class distribution
   - Examined sample images
   - Dropped duplicates

2. **Modeling**:
   - Custom CNN (2 Conv2D layers + dense)
   - Transfer learning using MobileNetV2 with frozen base
   - Normalized inputs and stratified split for class balance

3. **Results**:
   - CNN Accuracy: ~71%
   - MobileNetV2 Accuracy: ~83%
   - Submitted predictions to Kaggle

## How to Run

The full analysis and code can be found in `cancer_classification.ipynb`. A final `submission.csv` file is included.

## Submission

Final leaderboard score screenshot is in `kaggle_score.png`.

# Histopathologic Cancer Detection
### Problem Description and Dataset Overview

This project addresses a binary image classification problem using data from the Histopathologic Cancer Detection competition on Kaggle. The goal is to predict whether a small tissue image (96×96 pixels) contains metastatic cancer.

The dataset consists of:
- `train_labels.csv`: Contains 220,025 rows with two columns:
  - `id`: Image filename (without extension)
  - `label`: 0 = no cancer, 1 = cancer
- Image files: Each `.tif` image is a 96×96 pixel RGB image located in the `train/` and `test/` directories.
- `sample_submission.csv`: A template for submitting test predictions.

This is a medical imaging task with real-world relevance, requiring careful evaluation of class imbalance, image preprocessing, and effective use of limited computational resources.

### Exploratory Data Analysis (EDA)

EDA steps performed:

1. Duplicate Check:
   - Verified there were no duplicate rows in the dataset.

2. Class Balance:
   - The dataset has a slight imbalance:
     - ~59.5% labeled `0` (no cancer)
     - ~40.5% labeled `1` (cancer)
   - This was visualized using a count plot and addressed later using stratified sampling.

3. Image Properties:
   - Verified image dimensions (96×96×3), data type (`uint8`), and pixel intensity range (0–255).
   - Confirmed that images are RGB, not grayscale.

4. Visual Sampling:
   - Displayed 5 random images from each class to manually inspect visual features of cancer vs. non-cancer.

These steps informed our approach to balancing the dataset and choosing appropriate models.

### Model Architecture and Training Strategy

Two models were trained for comparison:

#### 1. Custom CNN
- A lightweight convolutional neural network designed for small image input:
  - Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output
  - Used `ReLU` activations and a final `sigmoid` for binary classification
  - Trained on a balanced 5,000-sample subset using stratified sampling
  - Normalized image pixel values to [0, 1]

#### 2. Transfer Learning with MobileNetV2
- A pretrained `MobileNetV2` model (`include_top=False`) used as a frozen feature extractor
- Image size increased to 128×128 and preprocessed using `preprocess_input()`
- Added custom head:
  - GlobalAveragePooling → Dense(64, relu) → Output(sigmoid)
- Trained on the same 5,000-sample subset (resized)

Both models were trained for 5 epochs with `Adam` optimizer and `binary_crossentropy` loss.

### Results and Evaluation

| Model        | Validation Accuracy | Validation Loss |
|--------------|---------------------|-----------------|
| Custom CNN   | ~71.4%              | ~0.58           |
| MobileNetV2  | ~82.9%              | ~0.37           |

Key takeaways:
- The pretrained model outperformed the custom CNN in both accuracy and generalization.
- Class imbalance and small sample size limited performance, but the transfer model still captured meaningful patterns.
- Training time was acceptable for both, but MobileNetV2 scaled better.

### Kaggle Submission

The best model (MobileNetV2) was used to generate predictions for the full test set.  
Each test image was resized, preprocessed, and passed through the model in batches.  
Predictions were thresholded at 0.5 and submitted to Kaggle in `submission.csv`.

### Conclusion and Reflections

- Transfer learning provided significant performance improvements over a basic CNN, especially with limited data.
- Data preprocessing and balancing were crucial to avoid biased learning.
- Future improvements could include:
  - Using a larger training sample
  - Data augmentation (rotation, flipping, zoom)
  - Fine-tuning pretrained layers
  - Adding dropout/regularization to avoid overfitting

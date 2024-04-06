# CHEST-HEALTH

#Classifying the X-Ray of human according to their disease using Support Vector Machines

## Overview
The project aims to classify X-Ray Images into various classes using a machine learning approach. It utilizes Support Vector Machines (SVMs) for classification tasks. The workflow includes data preprocessing, model training, evaluation, and prediction on test data.

## Setup and Usage

1. **Dataset Preparation**: 
   - Ensure you have a dataset containing X-ray images and corresponding labels. The images should be stored in a directory, and labels should be provided in a CSV file (`labels.csv`). Sample of how the data directory should look is given.

2. **Data Preprocessing**:
   - Run the provided code to preprocess the data. Adjust parameters such as image size, augmentation techniques, and batch size as needed. The class distribution is as follows:   
```python
Class distribution: Counter({'No Finding': 3044, 'Infiltration': 503, 'Effusion': 203, 'Atelectasis': 192, 'Nodule': 144, 'Pneumothorax': 114, 'Mass': 99, 'Consolidation': 72, 'Effusion|Infiltration': 69, 'Pleural_Thickening': 65, 'Atelectasis|Infiltration': 57, 'Atelectasis|Effusion': 55, 'Cardiomegaly': 50, 'Infiltration|Nodule': 44, 'Emphysema': 42, 'Edema': 41, 'Fibrosis': 38, 'Atelectasis|Effusion|Infiltration': 31, 'Cardiomegaly|Effusion': 30, 'Infiltration|Mass': 29, 'Edema|Infiltration': 21, 'Effusion|Pneumothorax': 20, 'Infiltration|Pneumothorax': 19, 'Consolidation|Infiltration': 18, 'Mass|Nodule': 17, 'Mass|Pneumothorax': 16, 'Effusion|Mass': 16, 'Emphysema|Pneumothorax': 15, 'Atelectasis|Consolidation': 15, 'Effusion|Pleural_Thickening': 14, 'Pneumonia': 14, 'Infiltration|Pleural_Thickening': 12, 'Consolidation|Effusion': 12, 'Atelectasis|Nodule': 12, 'Atelectasis|Consolidation|Effusion': 11, 'Effusion|Nodule': 11, 'Cardiomegaly|Infiltration': 10, 'Edema|Effusion|Infiltration': 10, 'Consolidation|Mass': 9, 'Mass|Pleural_Thickening': 9, 'Atelectasis|Pneumothorax': 9, 'Consolidation|Effusion|Infiltration': 8, 'Effusion|Emphysema': 8, 'Infiltration|Pneumonia': 8, 'Emphysema|Mass': 7, 'Nodule|Pneumothorax': 7, 'Emphysema|Infiltration': 6, 'Fibrosis|Infiltration': 6, 'Effusion|Infiltration|Nodule': 6, 'Nodule|Pleural_Thickening': 6, 'Cardiomegaly|Effusion|Infiltration': 6, 'Edema|Effusion': 6, 'Atelectasis|Effusion|Pneumothorax': 6, 'Pleural_Thickening|Pneumothorax': 6, 'Atelectasis|Mass': 5, 'Atelectasis|Pneumonia': 5, 'Atelectasis|Pleural_Thickening': 5, 'Atelectasis|Fibrosis': 5, 'Hernia': 5, 'Effusion|Mass|Nodule': 5, 'Effusion|Fibrosis': 5, 'Effusion|Emphysema|Pneumothorax': 5, 'Consolidation|Pleural_Thickening': 4, 'Fibrosis|Pleural_Thickening': 4, 'Atelectasis|Consolidation|Effusion|Infiltration': 4, 'Cardiomegaly|Edema': 4, 'Atelectasis|Consolidation|Infiltration': 4, 'Consolidation|Effusion|Mass': 4, 'Atelectasis|Cardiomegaly|Effusion': 4, 'Effusion|Infiltration|Pleural_Thickening': 4, 'Atelectasis|Edema': 4, 'Atelectasis|Effusion|Mass': 4, 'Atelectasis|Cardiomegaly': 4, 'Effusion|Infiltration|Pneumothorax': 4, 'Edema|Infiltration|Pneumonia': 4, 'Cardiomegaly|Emphysema': 3, 'Cardiomegaly|Consolidation': 3, 'Atelectasis|Infiltration|Pleural_Thickening': 3, 'Atelectasis|Emphysema': 3, 'Emphysema|Pleural_Thickening': 3, 'Effusion|Infiltration|Mass': 3, 'Mass|Nodule|Pneumothorax': 3, 'Consolidation|Pneumonia': 3, 'Atelectasis|Effusion|Emphysema|Infiltration': 3, 'Fibrosis|Mass': 3, 'Edema|Pneumonia': 3, 'Consolidation|Edema': 3, 'Consolidation|Nodule': 3, 'Cardiomegaly|Consolidation|Effusion': 3, 'Consolidation|Effusion|Infiltration|Nodule': 2, 'Atelectasis|Effusion|Infiltration|Mass': 2, 'Infiltration|Nodule|Pneumonia': 2, 'Effusion|Pleural_Thickening|Pneumothorax': 2, 'Nodule|Pneumonia': 2, 'Atelectasis|Fibrosis|Infiltration': 2, 'Infiltration|Nodule|Pleural_Thickening': 2, 'Atelectasis|Infiltration|Pneumothorax': 2, 'Mass|Nodule|Pleural_Thickening': 2, 'Atelectasis|Infiltration|Nodule': 2, 'Emphysema|Mass|Pneumothorax': 2, 'Atelectasis|Hernia': 2, 'Atelectasis|Emphysema|Mass': 2, 'Atelectasis|Consolidation|Effusion|Pleural_Thickening': 2, 'Effusion|Mass|Nodule|Pleural_Thickening': 2, 'Effusion|Nodule|Pneumothorax': 2, 'Consolidation|Infiltration|Mass': 2, 'Atelectasis|Effusion|Emphysema': 2, 'Effusion|Infiltration|Nodule|Pneumothorax': 2, 'Emphysema|Infiltration|Pleural_Thickening|Pneumothorax': 1, 'Cardiomegaly|Edema|Effusion': 1, 'Atelectasis|Infiltration|Mass|Pleural_Thickening': 1, 'Consolidation|Fibrosis': 1, 'Consolidation|Infiltration|Pneumothorax': 1, 'Edema|Fibrosis': 1, 'Emphysema|Infiltration|Pleural_Thickening': 1, 'Cardiomegaly|Effusion|Pneumonia': 1, 'Consolidation|Infiltration|Pneumonia': 1, 'Atelectasis|Effusion|Infiltration|Nodule': 1, 'Fibrosis|Nodule': 1, 'Effusion|Mass|Pleural_Thickening': 1, 'Atelectasis|Consolidation|Edema|Effusion|Infiltration': 1, 'Atelectasis|Effusion|Fibrosis': 1, 'Consolidation|Pneumothorax': 1, 'Atelectasis|Consolidation|Effusion|Emphysema': 1, 'Mass|Nodule|Pneumonia': 1, 'Atelectasis|Cardiomegaly|Effusion|Fibrosis|Infiltration': 1, 'Consolidation|Effusion|Pneumothorax': 1, 'Atelectasis|Cardiomegaly|Infiltration': 1, 'Cardiomegaly|Effusion|Fibrosis': 1, 'Consolidation|Infiltration|Nodule|Pneumothorax': 1, 'Emphysema|Pneumonia': 1, 'Atelectasis|Effusion|Mass|Nodule': 1, 'Atelectasis|Cardiomegaly|Effusion|Fibrosis|Nodule': 1, 'Emphysema|Infiltration|Nodule|Pneumothorax': 1, 'Hernia|Infiltration|Mass': 1, 'Edema|Effusion|Mass': 1, 'Infiltration|Pleural_Thickening|Pneumothorax': 1, 'Consolidation|Emphysema|Infiltration': 1, 'Consolidation|Fibrosis|Pneumothorax': 1, 'Fibrosis|Hernia|Mass': 1, 'Atelectasis|Consolidation|Mass|Pneumonia': 1, 'Atelectasis|Consolidation|Edema': 1, 'Atelectasis|Consolidation|Mass|Pneumothorax': 1, 'Infiltration|Mass|Nodule|Pneumothorax': 1, 'Emphysema|Fibrosis': 1, 'Atelectasis|Emphysema|Infiltration|Mass|Pneumothorax': 1, 'Atelectasis|Hernia|Pneumothorax': 1, 'Atelectasis|Effusion|Hernia': 1, 'Atelectasis|Consolidation|Edema|Infiltration|Pneumonia': 1, 'Atelectasis|Edema|Effusion|Infiltration|Pneumonia': 1, 'Edema|Infiltration|Mass|Pneumonia|Pneumothorax': 1, 'Atelectasis|Consolidation|Pneumothorax': 1, 'Atelectasis|Emphysema|Infiltration|Pneumothorax': 1, 'Fibrosis|Nodule|Pleural_Thickening': 1, 'Emphysema|Nodule|Pneumothorax': 1, 'Atelectasis|Emphysema|Pneumothorax': 1, 'Fibrosis|Pleural_Thickening|Pneumothorax': 1, 'Consolidation|Fibrosis|Infiltration': 1, 'Atelectasis|Consolidation|Infiltration|Pneumothorax': 1, 'Effusion|Mass|Pleural_Thickening|Pneumothorax': 1, 'Atelectasis|Emphysema|Infiltration': 1, 'Cardiomegaly|Pleural_Thickening': 1, 'Cardiomegaly|Fibrosis|Infiltration': 1, 'Atelectasis|Mass|Nodule': 1, 'Atelectasis|Nodule|Pneumothorax': 1, 'Effusion|Emphysema|Pleural_Thickening': 1, 'Atelectasis|Consolidation|Nodule': 1, 'Atelectasis|Consolidation|Effusion|Fibrosis|Pleural_Thickening': 1, 'Consolidation|Effusion|Infiltration|Pleural_Thickening': 1, 'Consolidation|Edema|Effusion|Pneumonia': 1, 'Edema|Mass': 1, 'Cardiomegaly|Edema|Infiltration|Pneumonia': 1, 'Fibrosis|Infiltration|Pleural_Thickening': 1, 'Hernia|Mass': 1, 'Atelectasis|Consolidation|Effusion|Infiltration|Pneumonia': 1, 'Cardiomegaly|Edema|Mass': 1, 'Atelectasis|Edema|Effusion': 1, 'Effusion|Fibrosis|Infiltration|Nodule': 1, 'Atelectasis|Effusion|Emphysema|Pneumothorax': 1, 'Effusion|Infiltration|Mass|Pneumothorax': 1, 'Cardiomegaly|Effusion|Nodule': 1, 'Atelectasis|Effusion|Nodule': 1, 'Edema|Infiltration|Mass': 1, 'Emphysema|Infiltration|Mass|Pneumothorax': 1, 'Atelectasis|Effusion|Mass|Pleural_Thickening': 1, 'Emphysema|Nodule': 1, 'Atelectasis|Effusion|Hernia|Infiltration': 1, 'Effusion|Fibrosis|Mass|Nodule': 1, 'Effusion|Mass|Nodule|Pneumothorax': 1, 'Edema|Emphysema|Infiltration|Pneumonia': 1, 'Cardiomegaly|Consolidation|Effusion|Mass|Pneumothorax': 1, 'Consolidation|Effusion|Infiltration|Mass': 1, 'Atelectasis|Cardiomegaly|Effusion|Mass': 1, 'Consolidation|Effusion|Mass|Nodule|Pleural_Thickening': 1, 'Atelectasis|Consolidation|Infiltration|Mass|Pleural_Thickening': 1, 'Mass|Pleural_Thickening|Pneumothorax': 1, 'Consolidation|Effusion|Pleural_Thickening': 1, 'Atelectasis|Consolidation|Effusion|Mass|Nodule|Pneumothorax': 1, 'Atelectasis|Cardiomegaly|Consolidation|Effusion|Infiltration|Mass|Pleural_Thickening': 1, 'Atelectasis|Cardiomegaly|Consolidation': 1, 'Consolidation|Effusion|Infiltration|Mass|Nodule': 1, 'Atelectasis|Consolidation|Infiltration|Pneumonia': 1, 'Fibrosis|Infiltration|Mass': 1, 'Atelectasis|Consolidation|Nodule|Pleural_Thickening|Pneumothorax': 1, 'Effusion|Infiltration|Mass|Nodule': 1, 'Atelectasis|Emphysema|Fibrosis|Pleural_Thickening': 1, 'Consolidation|Infiltration|Mass|Nodule': 1, 'Atelectasis|Edema|Effusion|Pneumonia': 1, 'Atelectasis|Effusion|Pleural_Thickening|Pneumothorax': 1, 'Consolidation|Mass|Nodule|Pneumonia': 1, 'Infiltration|Mass|Nodule|Pleural_Thickening|Pneumothorax': 1, 'Atelectasis|Consolidation|Effusion|Emphysema|Nodule|Pneumothorax': 1, 'Edema|Effusion|Nodule': 1, 'Infiltration|Mass|Nodule|Pleural_Thickening': 1, 'Cardiomegaly|Effusion|Infiltration|Nodule': 1, 'Atelectasis|Cardiomegaly|Effusion|Infiltration|Pleural_Thickening': 1, 'Pleural_Thickening|Pneumonia': 1, 'Edema|Infiltration|Nodule': 1, 'Effusion|Nodule|Pleural_Thickening': 1, 'Consolidation|Edema|Infiltration': 1, 'Effusion|Emphysema|Infiltration': 1, 'Consolidation|Infiltration|Pleural_Thickening': 1, 'Edema|Nodule': 1, 'Cardiomegaly|Consolidation|Pneumonia': 1, 'Atelectasis|Edema|Effusion|Infiltration': 1, 'Consolidation|Emphysema': 1, 'Cardiomegaly|Effusion|Pleural_Thickening': 1, 'Effusion|Pneumonia': 1, 'Emphysema|Pneumonia|Pneumothorax': 1, 'Infiltration|Mass|Nodule': 1, 'Effusion|Infiltration|Pneumonia': 1, 'Atelectasis|Emphysema|Fibrosis': 1, 'Atelectasis|Effusion|Pleural_Thickening': 1, 'Atelectasis|Emphysema|Infiltration|Nodule|Pneumothorax': 1, 'Cardiomegaly|Effusion|Mass|Pneumothorax': 1, 'Cardiomegaly|Edema|Infiltration|Nodule': 1, 'Atelectasis|Effusion|Infiltration|Pneumothorax': 1, 'Atelectasis|Consolidation|Mass': 1, 'Cardiomegaly|Mass': 1, 'Cardiomegaly|Consolidation|Infiltration': 1, 'Emphysema|Pleural_Thickening|Pneumothorax': 1, 'Atelectasis|Consolidation|Effusion|Mass': 1, 'Consolidation|Effusion|Pneumonia': 1, 'Cardiomegaly|Effusion|Emphysema': 1})

```

3. **Model Training**:
   - Train an SVM model on the preprocessed data. The model is trained using the flattened image vectors obtained after preprocessing.

4. **Evaluation**:
   - Evaluate the trained model on the test data to assess its performance. Metrics such as accuracy, classification report, and confusion matrix are computed.

5. **Prediction**:
   - Make predictions on new or unseen test data using the trained SVM model.

## Requirements

- Python 3
- TensorFlow
- NumPy
- OpenCV
- scikit-learn
- Matplotlib
- pandas

## Usage Example

```python
# Ensure necessary imports and setup are done before executing

# Run data preprocessing
preprocess_and_save_data(image_folder, labels_file, batch_size, output_folder)

# Train the SVM model
svm_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Make predictions on test data
y_pred_test = svm_model.predict(X_test_flat)

# Evaluate model performance
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

```
## Acknowledgements
The code in this repository is based on work by Manya Joshi.
The dataset used in this project is sourced from Ramdom Sample of NIH chest dataset[https://www.kaggle.com/datasets/nih-chest-xrays/sample]

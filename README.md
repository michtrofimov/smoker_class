# Smoker Status Classification in Lung Adenocarcinoma RNA-seq

This repository contains the code and data used for classifying smoker status in Lung Adenocarcinoma (LUAD) using RNA-seq data and machine learning techniques.

## Project Overview

Lung cancer is the leading cause of cancer-related deaths worldwide, with smoking accounting for almost 85% of lung cancer cases. Classifying smoker status can aid in early diagnosis and improve the accuracy of lung cancer diagnosis. Gene expression can be altered due to smoking and is therefore a potential biomarker for lung cancer.

## Objectives

1. **Dataset Collection**: Identify datasets of RNA-seq with adequate smoking annotation.
2. **Algorithm Selection**: Find the best performing machine learning algorithm for classification.
3. **Model Development**: Write a hierarchical multiclassifier.

## Methods and Data

### Dataset

- **TCGA Lung Adenocarcinoma Dataset**: Comprised of 522 patients.

### Tools and Libraries

- **Python Libraries**:
  - Machine Learning: `scikit-learn`, `LightGBM`, `pycaret`, `optuna`
  - Feature Engineering: `SHAP`, `decoupleR`
- **R Libraries**:
  - Pathway Analysis: `PROGENY`

### Data Preparation

- **Data Transformation**: Multiclass Y transformed to binary Y.
- **Gene Filtration with SHAP**: Estimated feature importance using SHAP values and removed the least important features iteratively.

### Model Training

- **Initial Model Assessment**:
  - Performance estimation using `pycaret` on the training dataset (396 samples, 3000 genes).
  - Best model: LightGBM with F1 score ~ 0.75.
  
- **Hyperparameter Tuning**:
  - Hyper-tuned LightGBM using `Optuna`.
  - Train F1: 1.00, Test F1: 0.73.
  
- **Pathway Activities**:
  - Assessed pathway activities with `PROGENY`, resulting in 14 new features.
  - Train F1: 0.91, Test F1: 0.74.

<img src="/home/m_trofimov/BI/project/figures/Screenshot 2024-05-26 at 21.10.40.png" alt="Project Overview" width="600">


### Multiclassification

- **Custom Hierarchical Classifier**:
  - Represented multiclass Y as a tree hierarchy.
  - Implemented a Custom Hierarchical Classifier.
  - Upsampled classes with `SMOTE`.
  - Trained and hyper-tuned the LightGBM multiclassifier.

## Future Plans

1. Identify additional RNA-seq LUAD datasets.
2. Explore deep learning algorithms.
3. Improve the Custom Hierarchical Multiclassifier.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/michtrofimov/smoker_class.git
   cd smoker_class
   ```
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Notebooks**:
    Open the Jupyter notebooks in the notebooks/ directory to see data preprocessing steps, model training, and evaluation.

## Contact
For any questions or issues, please contact [Michil Trofimov](https://github.com/michtrofimov).
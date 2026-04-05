# Sentiment Analysis on Social Media Posts (X)

---

## Overview

This project implements a complete **machine learning pipeline for sentiment classification** of social media posts (X/Twitter-like platform).

The goal is to classify posts as **Positive** or **Negative** using textual content alongside user and engagement metadata.

> Developed as part of a course on **machine learning**, which covered topics such as supervised & unsupervised learning, feature engineering & selection, and model evaluation. April-July 2025.

---

## Dataset

- ~40,000 labeled social media posts  
- Binary sentiment labels: **Positive / Negative**  
- Includes real-world features:
  - Text content  
  - User metadata  
  - Engagement metrics (likes, retweets)  
  - Temporal features (time, day, timezone)

### Files:
- `sentiment.csv` - Main dataset used in the project (used to split into training and evaluation sets)
- `test_predictions.csv` - Predictions generated on the [test set](\test.csv) (labels were not disclosed during the project)
- `test.csv` - Test dataset.

---

## Project Workflow

### Phase 1: Exploratory Data Analysis & Feature Engineering

**Key insights from EDA:**
- Slight class imbalance (~56.5% negative, ~43.5% positive)
- Long-tail distributions in likes & retweets
- Sentiment varies with:
  - Time of day  
  - Content type (post/comment/share)  
  - Account characteristics  

**Data preprocessing:**
- Outlier removal using **IQR**
- Missing values handling:
  - Numerical → Median / KNNImputer  
  - Categorical → Distribution-based imputation  
- Removed samples with missing labels  

**Feature engineering (110+ features):**
- **Temporal features**: day of week, time buckets  
- **User behavior**: account age, activity rates  
- **Engagement ratios**: likes/retweets, likes/followers  
- **Text features**: word count, hashtags, TF-IDF (with N-grams)  
- **Combined features**: cross-feature interactions  

---

### Feature Selection

- Applied **Fisher Score (Filter Method)**  
- Reduced dimensionality from **110 → 22 features**  
- Retained features with the highest class discrimination power  

---

### Phase 2: Modeling & Evaluation

**Models explored:**
- Decision Trees (optimized with Grid Search)  
- Neural Networks (MLP)  
- Support Vector Machines (SVM)  
- Random Forest (final model)  

**Validation strategy:**
- **Stratified 10-Fold Cross Validation**
- Focus on **Recall** and **AUC-ROC**
- Ensures robustness and reduces overfitting  

---

## Final Model Performance

The final selected model achieved strong and balanced performance:

- **Balanced Accuracy (BACC):** 0.925  
- **True Positive Rate (TPR / Recall):** 0.897  
- **False Positive Rate (FPR):** 0.0466  

---

## Key Insights

- Text-based features are the strongest predictors of sentiment  
- Engagement metrics add meaningful predictive signal  
- Temporal patterns influence sentiment distribution  
- Combining heterogeneous features significantly improves performance  

---

## Repository Structure
```
├── data/
│ ├── sentiment.csv # Main dataset (train + evaluation split)
│ ├── test_predictions.csv # Predictions on hidden test set
├── notebooks/
│ ├── Exploratory Data Analysis & Feature Engineering.ipynb # EDA, preprocessing, feature engineering
│ ├── Modeling & Evaluation.ipynb # Model training, tuning, evaluation
├── reports/
│ ├── Exploratory Data Analysis & Feature Engineering.pdf # Detailed analysis and discussion (Phase 1)
│ ├── Modeling & Evaluation.pdf # Detailed analysis and discussion (Phase 2)
```
---
## How to Run

This project is designed for execution in [Google Colab](https://colab.research.google.com/), including built-in file upload handling.
---

### 1. Open the Scripts

Upload the `.ipynb` files to a code editor of your choice or open them directly in Google Colab:

- **Part 1:** `Exploratory Data Analysis & Feature Engineering.ipynb`  
  (Data sensing, preprocessing, and feature engineering)

- **Part 2:** `Modeling & Evaluation.ipynb`  
  (Model training, tuning, and evaluation)

---

### 2. Upload the Dataset

When running the designated data loading cell in **Part 1**, a file upload prompt will appear using:

```python
google.colab.files.upload()
```

### 3. Select and upload the `sentiment.csv` file from your local machine
The script will remove invalid entries, begin the preprocessing and feature engineering pipeline

---
### 4. Output
- Final trained models and evaluation results
- `test_predictions.csv` containing predictions on the test set
---
---

## Contributors
- [Leah Rosen](https://github.com/leahrosen21)
- [Ofri Ratinsky](https://github.com/OfriRatinsky)


# Sentiment Analysis on Social Media Posts (X)

---

## Overview

This project implements a complete **machine learning pipeline for sentiment classification** of social media posts (X / Twitter-like platform).

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
- `sentiment.csv` — Main dataset used in the project (used to split into training and evaluation sets)
- `test_predictions.csv` — Predictions generated on the hidden test set (samples were not disclosed during the project)

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
├── sentiment.csv # Main dataset (train + evaluation split)
├── test_predictions.csv # Predictions on hidden test set
├── notebooks/
├── Exploratory Data Analysis & Feature Engineering.ipynb # EDA, preprocessing, feature engineering
├── Modeling & Evaluation.ipynb # Model training, tuning, evaluation
├── reports/
│ ├── Exploratory Data Analysis & Feature Engineering.pdf # Detailed analysis and discussion (Phase 1)
│ ├── Modeling & Evaluation.pdf # Detailed analysis and discussion (Phase 2)
```


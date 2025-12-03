# 🔍 Credit Card Fraud Detection Pipeline

This project demonstrates an end-to-end Machine Learning pipeline for detecting fraudulent credit card transactions.  
It covers data preprocessing, class imbalance handling, model training, evaluation, and explainability with SHAP.

The goal of this project is to showcase skills relevant for an **Applied Scientist** role:
- Working with highly imbalanced datasets  
- Building complete ML pipelines  
- Evaluating classification models using appropriate metrics  
- Interpreting model predictions and feature importance  
- Communicating results clearly and visually  

Dataset: **Kaggle – Credit Card Fraud Detection (284,807 transactions, 492 fraud cases)**  
## 📊 Dataset Description

The dataset contains **284,807 credit card transactions**, of which only **0.172% are fraudulent** — a highly imbalanced real-world scenario.

### 🔐 Features
- Columns **V1–V28**: PCA-transformed numerical features  
- **Time**: Seconds elapsed between each transaction  
- **Amount**: Transaction amount  
- **Class**:
  - `0` → legitimate transaction  
  - `1` → fraud (positive class)

### ⚠️ Problem Characteristics
- **Severe class imbalance**  
  Fraud cases represent less than **0.2%** of all transactions.
- **Sensitive financial domain**  
  Requires metrics beyond accuracy (F1, recall, precision).
- **Goal**  
  Build a model that maximizes fraud detection while controlling false positives.
## 🗂️ Project Structure

```bash
fraud-detection-pipeline/
│── 📁 data/ # (ignored in .gitignore) raw dataset
│── 📁 models/ # saved models (ignored)
│── 📁 plots/ # evaluation and SHAP plots
│── 📄 fraud_detection.ipynb # full notebook (end-to-end pipeline)
│── 📄 requirements.txt # dependencies
│── 📄 README.md # project documentation
```
### 📌 Notes
- Large files (`data/creditcard.csv`, saved models) are **not pushed** to GitHub.  
  They regenerate automatically when running the notebook.
- All results, metrics, and visualizations are produced directly inside the notebook.
## 🎯 Project Objectives

This project demonstrates an end-to-end **Fraud Detection Machine Learning Pipeline**, designed to show strong applied-scientist skills through:

### 🔹 1. **Real-World Imbalanced Classification**
Credit card fraud datasets are heavily imbalanced.  
The goal is to build a model that correctly identifies rare fraudulent transactions while minimizing false alarms.

### 🔹 2. **Complete ML Workflow**
The project includes:
- exploratory data analysis (EDA)
- preprocessing & scaling
- train/validation split
- handling class imbalance with Stratified K-Fold & class weights
- training a Logistic Regression baseline
- evaluation with ROC-AUC, Precision, Recall, F1
- interpretability with SHAP

### 🔹 3. **Industry-Grade Explainability**
SHAP is used to show:
- global importance of features  
- how individual attributes influence fraud predictions  
This is crucial for financial institutions and aligns with Microsoft Applied Scientist interview expectations.

### 🔹 4. **Clean, Modular, Reproducible Notebook**
All pipeline steps are structured clearly and can be extended into:
- XGBoost / LightGBM models  
- automated feature engineering  
- real-time API scoring  

This project demonstrates your ability to solve practical ML problems using best practices.
## 📊 Dataset Description

The project uses the **Credit Card Fraud Detection Dataset** (Kaggle), which contains real transactions made by European cardholders in 2013.

### 📁 Dataset Overview
- **Total samples:** 284,807  
- **Fraud cases:** 492  
- **Fraud ratio:** 0.172% (highly imbalanced)

### 📌 Feature Details
- All input features (`V1`–`V28`) are PCA-transformed for anonymity.
- `Time` — seconds elapsed between this transaction and the first recorded transaction.
- `Amount` — transaction amount.
- `Class` — target label  
  - `0` → legitimate  
  - `1` → fraud  

### ⚠️ Dataset Storage
The dataset is **not included in the repository** due to size.  
Instead, it is loaded manually by the user and placed in:

```bash
07-fraud-detection-pipeline/data/creditcard.csv
```

This path is ignored via `.gitignore`, keeping the repository lightweight and professional.
## 🛠️ Environment & Installation

Follow these steps to set up the environment and run the fraud detection pipeline.

---

### 🔧 1. **Create and Activate Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```
## 🛠️ Environment & Installation

### 📦 Install Dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```
### 📚 Key Libraries

- **pandas** — data loading & preprocessing  
- **numpy** — numerical computation  
- **scikit-learn** — modeling, scaling, evaluation  
- **seaborn** / **matplotlib** — visualizations  
- **imbalanced-learn** — handling class imbalance  
- **shap** — explainability  

---

### 📁 Add the Dataset

Download the **Credit Card Fraud Detection** dataset (Kaggle) and place it in:

```bash
07-fraud-detection-pipeline/data/creditcard.csv
```

⚠️ This dataset folder is ignored in `.gitignore`, so it is **not included in the repository**.

---

### ▶️ Run the Notebook

Open the main notebook:

```bash
fraud_detection.ipynb
```

Run all cells to execute the full ML pipeline:

- data preprocessing  
- scaling and train/validation split  
- handling imbalance with class weights  
- model training (Logistic Regression baseline)  
- evaluation metrics (Recall, Precision, F1, ROC-AUC)  
- generating feature importance and SHAP interpretability plots  

This notebook reproduces the entire experiment end-to-end.
## 🔍 7. Exploratory Data Analysis (EDA)

Before building the fraud detection model, we analyze the dataset to understand patterns, imbalance, and feature behavior.

### 📊 What We Explore

#### 1️⃣ Class Distribution (Fraud vs Non-Fraud)
The dataset is highly imbalanced:
- **0 → Legitimate transactions**
- **1 → Fraudulent transactions (only 0.172%)**

Understanding imbalance is critical because accuracy alone becomes meaningless.  
Instead, we focus on:
- **Recall** (catching fraud)
- **Precision** (avoiding false alarms)

---

#### 2️⃣ Transaction Amount Distribution
Fraudulent transactions often differ in amount distribution patterns.  
Visualization highlights:
- the typical amounts for non-fraud  
- unusual spikes for fraud cases  

---

#### 3️⃣ Correlation Heatmap
Although most features are PCA components (`V1–V28`), correlations may still reveal:
- feature clusters  
- redundancies  
- relationships useful for model interpretability  

---

#### 4️⃣ Feature Histograms
Visual inspection of individual components helps understand:
- which PCA features separate fraud  
- possible outliers  
- feature spread

---

### 🧠 Why EDA Matters
Performing EDA ensures we:
- understand the imbalance problem  
- choose proper evaluation metrics  
- prepare valid preprocessing steps  
- detect data leakage or anomalies  
- build intuition for model behavior  

This step sets the foundation for the modeling pipeline that follows.
## 🧹 8. Preprocessing & Feature Engineering

Before training the fraud detection model, we apply several preprocessing steps to ensure data quality and model stability.

---

### 🔧 1️⃣ Load & Inspect the Dataset
We begin by loading the dataset, checking dimensions, missing values, and basic statistics.  
The dataset has **no missing values** and contains standardized PCA-transformed features (`V1–V28`).

---

### ⚖️ 2️⃣ Handling Class Imbalance
Fraud cases represent only **0.172%** of the dataset.  
To address this extreme imbalance, we use:

- **Stratified train/test split** → preserves fraud ratio  
- **Class weights** in Logistic Regression → penalizes misclassifying fraud  
- *(Optional)* SMOTE / undersampling  
  *(Not used here to avoid synthetic distortion)*  

These techniques ensure the model focuses on catching rare fraud cases.

---

### 📏 3️⃣ Feature Scaling
We apply **StandardScaler** to numeric features.

Reasons:
- Logistic Regression is sensitive to feature magnitude  
- PCA components vary in scale  
- Scaling improves convergence and stability  

Scaling is fitted on training data only → preventing data leakage.

---

### 🧪 4️⃣ Train / Validation Split
We use a **stratified 80/20 split**:

- Training set → model fitting  
- Validation set → performance evaluation  

This ensures identical fraud distribution across splits.

---

### 🚫 5️⃣ Data Leakage Check
Before modeling, we ensure:
- No target leakage  
- No duplicate transactions  
- No engineered features derived from fraud labels  

The dataset is clean and safe for modeling.

---

### 🎯 Summary
After preprocessing, the dataset is:
- scaled  
- balanced using class weights  
- split into training and validation sets  
- ready for model training  

This prepares the foundation for a reliable fraud detection model.
## 🤖 9. Model Training & Evaluation

After preparing the dataset, we train a baseline fraud detection model and evaluate its performance using metrics that matter for imbalanced classification.

---

### 🧠 1️⃣ Model Selection: Logistic Regression

We start with a **Logistic Regression** classifier because it is:

- fast and interpretable  
- performs well on standardized PCA features  
- robust with class weights  
- easy to explain using SHAP  

This model serves as a strong baseline before trying more complex algorithms (e.g., XGBoost, LightGBM).

---

### ⚖️ 2️⃣ Class Weights to Handle Imbalance

Fraud cases are extremely rare, so we use:

```bash
class_weight = "balanced"
```

This increases the penalty for misclassifying fraud, improving Recall and F1-score.

---

### 🚀 3️⃣ Model Training

Training is performed on the **scaled training set**.  
The model learns to separate legitimate vs fraudulent transactions based on PCA-transformed features.

---

### 📊 4️⃣ Evaluation Metrics

Since accuracy is useless for highly imbalanced data, we rely on:

- **Recall (Sensitivity)** → ability to detect fraud  
- **Precision** → avoiding false positives  
- **F1-Score** → balance between Precision & Recall  
- **ROC-AUC** → ranking ability across thresholds  
- **Confusion Matrix** → detailed classification breakdown  

These metrics reflect real-world financial fraud detection priorities, where catching fraud (high Recall) is essential.

---

### 🧾 5️⃣ Key Results (Typical)

You should see results similar to:

- **Recall (Fraud)**: ~0.85+  
- **Precision (Fraud)**: ~0.20–0.30  
- **F1-Score**: ~0.30–0.40  
- **ROC-AUC**: ~0.95  

This pattern is normal due to extreme imbalance.  
High Recall is prioritized — better to investigate more cases than miss fraudulent ones.

---

### 🎯 Summary

Our trained model:

- correctly identifies a large portion of fraud cases  
- maintains a balanced F1-score  
- achieves high ROC-AUC  
- provides a solid foundation for interpretability with SHAP  

Next, we analyze *why* the model makes its predictions using feature importance and global explainability techniques.
## 🧠 10. Model Explainability & SHAP

Understanding *why* a model predicts “fraud” is essential in financial systems.  
This section provides interpretability using:

- **Global Feature Importance**
- **SHAP Value Analysis**
- **Transaction-Level Explanation (Local SHAP)**

These methods help analysts, auditors, and regulators trust model behavior.

---

### 🔍 10.1 Global Feature Importance

We analyze the importance of each PCA-transformed feature by calculating:

- Average absolute SHAP values per feature  
- Ranking features based on global contribution  

This helps identify which components influence fraud detection the most.

Typical insights:

- Some PCA components dominate fraud prediction  
- The distribution of importance is highly skewed  
- Feature importance aligns with patterns seen during EDA  

A bar chart is generated to visualize the top contributing features.

---

### 🐝 10.2 SHAP Summary Plot Alternative (Global Overview)

Because PCA components and SHAP dimensions may not perfectly align for classic beeswarm plotting,  
we instead use a **global SHAP bar chart** to show:

- mean absolute SHAP value for each feature  
- ranking of top predictors  
- overall contribution distribution  

This provides a clean, interpretable global explanation without dimensionality issues.

---

### 🧾 10.3 Local Explanation (Single Transaction)

For any individual transaction, we compute local SHAP values to visualize:

- which features push the prediction *toward fraud*  
- which features push it *toward non-fraud*  

This produces a waterfall-style interpretation, e.g.:

- large positive SHAP values → increase fraud likelihood  
- large negative SHAP values → decrease fraud likelihood  

Such explainability is required in fields like:

- banking  
- anomaly detection  
- compliance & auditing  
- security incident analysis  

---

### 🎯 Summary

SHAP analysis provides:

- **global understanding** of key drivers behind fraud  
- **local transparency** for individual flagged transactions  
- compliance-ready explanations for real-world deployment  

This makes the model not only effective but also **trustworthy** and **interpretable**.
## 🏁 11. Conclusions & Next Steps

### ✅ Key Takeaways

This project demonstrates a complete, industry-grade **fraud detection pipeline**, covering:

- rigorous preprocessing and handling of extreme class imbalance  
- interpretable modeling with Logistic Regression  
- evaluation using meaningful metrics (Recall, Precision, F1, ROC-AUC)  
- global and local explainability using SHAP  

Despite the challenge of detecting rare fraudulent cases, the model achieves:

- **High Recall**, ensuring most fraud cases are caught  
- **Strong ROC-AUC**, showing solid ranking capability  
- **Clear interpretability**, essential for financial institutions  

This aligns perfectly with expectations for **Applied Scientist** roles—combining ML theory, practical implementation, and explainability.

---

## 🚀 12. Next Steps (Potential Improvements)

Here are recommended enhancements if you wish to expand the project further:

### 🔹 1. Try Advanced Models
- XGBoost  
- LightGBM  
- Random Forest  
- Neural networks (MLP)

These models often outperform logistic regression on structured data.

---

### 🔹 2. Hyperparameter Tuning
Use:
- Grid Search  
- Random Search  
- Optuna / Bayesian Optimization  

Goal: maximize Recall while maintaining Precision balance.

---

### 🔹 3. Threshold Optimization
Instead of default threshold = 0.5, tune decision boundary for optimal F1 or Recall.

---

### 🔹 4. SMOTE Variants
Experiment with:
- SMOTE  
- ADASYN  
- SMOTEENN  

Useful if the PCA components behave well with synthetic sampling.

---

### 🔹 5. Model Monitoring
Track:
- drift in transaction distributions  
- sudden spikes in fraud activity  
- model performance over time  

Important for production systems.

---

## 🎉 Final Note

This project showcases your ability to build:

- a **clean**,  
- **structured**,  
- **fully reproducible**,  
- **explainable**  


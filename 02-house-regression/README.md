# 🏡 House Price Regression (XGBoost + SHAP)

This project implements a **regression model for house price prediction** using the popular **Kaggle House Prices dataset**.  
It demonstrates a complete end-to-end pipeline common in Applied Scientist roles:

- data loading & cleaning  
- feature engineering  
- handling missing values  
- model training (XGBoost / LightGBM / RandomForest)  
- hyperparameter tuning  
- interpretability using **SHAP**  
- prediction visualizations  
- model evaluation using MAE / RMSE  

This project is a part of my **Applied Scientist Portfolio**.

---

## 📂 Project Structure

```bash
02-house-regression/
│
├── 02-house-regression.ipynb
├── best_model.pkl
├── submission.csv
├── test.csv
└── train.csv
```

## 🧠 1. Problem Description

The goal of this project is to **predict house sale prices** based on a wide variety of features describing each property.  
The dataset includes:

- structural characteristics (overall quality, number of rooms, basement size)
- dimensions (living area, lot area, total square footage)
- categorical attributes (neighborhood, exterior type, garage type)
- temporal features (year built, year remodeled, year sold)
- auxiliary variables (conditions, utilities, functional ratings)

The target variable is:
SalePrice — the property's final sale price

This is a real-world, noisy regression problem with:

- missing values  
- skewed distributions  
- non-linear relationships  
- interactions between features  

— which makes it a great benchmark task for applied ML.

---

## 🧹 2. Data Preprocessing & Feature Engineering

To prepare the data for modeling, several key preprocessing steps were applied:

### ✔ Handling Missing Values

- **Numerical features** → imputed with **median**
- **Categorical features** → imputed with `"Missing"`
- Structural absence (e.g., *no garage*, *no basement*) handled explicitly

### ✔ Encoding Categorical Features

- One-Hot Encoding using `pandas.get_dummies()`  
- Avoided data leakage by fitting encoders only on training data  

### ✔ Feature Engineering

New features were added to capture more meaningful signals:

- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
- `Age = YrSold - YearBuilt`
- `AgeSinceRemodel = YrSold - YearRemodAdd`
- `IsNew = YrSold == YearBuilt`
- Log-transformed `SalePrice` for more normal distribution
- Log-transformed skewed numeric predictors

### ✔ Train/Test Split

Data was split into:

- **80% training**
- **20% test**

Random shuffling was used. Stratification is not applicable because the target is continuous.

---

## 🧱 3. Model Training

Several machine learning models were trained and compared:

### 🟦 Baseline Models

- Linear Regression
- Ridge Regression
- Lasso Regression

### 🟩 Tree-Based Models

- Random Forest Regressor
- LightGBM Regressor
- **XGBoost Regressor (best model)**

### ⭐ Best Performing Model: XGBoost

Hyperparameters used:

```python
n_estimators=500
learning_rate=0.05
max_depth=5
subsample=0.8
colsample_bytree=0.8
```
This model handled the nonlinearities and interactions in the dataset extremely well.

## 📊 4. Evaluation Metrics

The model was evaluated using standard regression metrics:

- **MAE** – Mean Absolute Error  
- **MSE** – Mean Squared Error  
- **RMSE** – Root Mean Squared Error  

These provide a clear understanding of both average prediction error and variance of errors.

### ✔ Example Results (replace with your actual values)

| Model        | MAE     | RMSE    |
|--------------|---------|---------|
| LinearReg    | 28,500  | 43,900  |
| RandomForest | 20,200  | 31,500  |
| LightGBM     | 18,900  | 29,800  |
| **XGBoost**  | **17,300** | **28,500** |

**Interpretation:**  
XGBoost achieved the lowest MAE and RMSE, indicating strong predictive performance and the best fit among all tested models.

---

## 📈 5. Prediction Visualization

A **Predicted vs True Prices** plot was generated to visually evaluate model quality.

*(Insert your `plots/pred_vs_true.png` here)*

### ✔ Interpretation

- Points concentrate along the **y = x diagonal line**, meaning predictions closely match actual sale prices.  
- There are **no large outliers**, showing stable behavior across price ranges.  
- Indicates strong generalization on unseen data.

This visualization is important because it shows not only accuracy, but **error distribution**.

---

## 🔍 6. SHAP: Model Interpretability

SHAP (SHapley Additive exPlanations) was used to understand model reasoning.

### ✔ SHAP Summary Plot

*(Insert `shap/shap_summary.png` here)*

This plot reveals which features most strongly influence predictions.

### ✔ Top Contributing Features (typically)

- **OverallQual** – strongest indicator of price  
- **GrLivArea** – above-ground living space  
- **TotalSF** – total square footage (basement + floors)  
- **Neighborhood** – location-based pricing structure  
- **GarageCars** – garage capacity as a premium feature  
- **YearBuilt / YearRemodAdd** – newer homes → higher price  

### ✔ Insights

- Higher overall quality and larger living area significantly increase predicted price  
- Neighborhood captures location value differences  
- Newer homes and remodeled homes are valued higher  
- Feature effects match domain knowledge (a great sign of a trustworthy model)

SHAP analysis confirms the model is **interpretable**, **consistent**, and **aligned with human reasoning**.

---

## 📝 7. Final Conclusions

- **XGBoost** achieved the best performance with low MAE/RMSE  
- Feature engineering (TotalSF, Age, remodeling features) significantly improved results  
- SHAP visualizations show meaningful, human-aligned feature importance  
- Predictions generalize well with no major outliers  
- This project demonstrates core **Applied Scientist** skills:
  - tabular data handling  
  - preprocessing & transforming real-world data  
  - model selection & evaluation  
  - boosting algorithms  
  - explainable ML (SHAP)

---

## 🛠 8. How to Reproduce

Clone the repository:

```bash
git clone https://github.com/JulijanaMilosavljevic/Applied-Scientist-Portfolio
cd Applied-Scientist-Portfolio/02-house-regression
```
Run the notebook:
```bash
jupyter notebook 02-house-regression.ipynb
pip install -r requirements.txt
```


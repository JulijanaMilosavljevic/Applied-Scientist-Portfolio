# 📊 01‑Tabular Churn

## 🎯 Project Overview
This project analyses the **Telco Customer Churn** dataset from Kaggle to predict **customer churn** (i.e., whether a customer will cancel their service).  
It demonstrates a full machine learning workflow: data cleaning, feature engineering, model training (three different models), evaluation, error analysis, and model export.

---

## 🧩 Contents
- **Data preparation**: convert data types, handle missing values, encode categorical features  
- **Three modelling approaches**:  
  - Logistic Regression (with imputation and scaling)  
  - Random Forest (tree-based model)  
  - XGBoost (gradient boosting)  
- **Evaluation & metrics**: accuracy, precision, recall, F1-score, and error analysis (confusion matrix)  
- **Model export**: saving the best scenario into a `.pkl` file for reuse or deployment

---

## 🚀 Project Pipeline
1. **Load dataset**  
2. **Clean & preprocess data**  
   - Convert `TotalCharges` to numeric  
   - Fill or impute missing values  
   - Encode categorical columns  
3. **Split data** into training and test sets  
4. **Define features**  
   - Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`  
   - Categorical: `gender`, `InternetService`, `PaymentMethod`, etc.  
5. **Build pipelines & train models**  
   - Logistic Regression with imputer + scaler  
   - Random Forest  
   - XGBoost  
6. **Evaluate models** on test set  
   - Classification report  
   - Confusion matrix for error analysis  
7. **Export best model** to `models/best_model.pkl`  
8. **Load exported model** and verify metrics

---

## 📁 Folder Structure
01-tabular-churn/ <br>
 ├── 01-tabular-churn.ipynb            
 ├── results/          
 ├── models/             
 ├── Telco-Customer-Churn.csv           
 └── README.md     

---

## ✅ Usage and How to Run
1. Clone the repository:
   git clone https://github.com/JulijanaMilosavljevic/Applied-Scientist-Portfolio.git

2. Navigate to the project folder:
   cd Applied-Scientist-Portfolio/01-tabular-churn

3. (Optional) Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

4. Open the Jupyter notebook and run it end-to-end.

5. Metrics and models will be saved in `results/` and `models/` folders.



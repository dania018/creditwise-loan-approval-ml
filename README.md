# CreditWise – Intelligent Loan Approval Prediction System

An end-to-end machine learning pipeline that predicts whether a loan application should be **Approved** or **Rejected**, built to help a simulated bank (SecureTrust Bank) replace slow, inconsistent manual loan verification with a faster, data-driven decision system.

## 📌 Problem Statement

SecureTrust Bank offers personal and home loans across urban and rural India. Loan officers currently evaluate applications manually, which is time-consuming and inconsistent — leading to two costly outcomes:
- Good applicants get rejected → lost business
- High-risk applicants get approved → financial losses

This project builds a supervised ML model that learns patterns from historical loan data to predict approval outcomes, supporting (not replacing) human decision-making.

## 📊 Dataset

- **1,000 applicant records**, 20 columns
- Features include: `Applicant_Income`, `Coapplicant_Income`, `Employment_Status`, `Age`, `Marital_Status`, `Dependents`, `Credit_Score`, `Existing_Loans`, `DTI_Ratio`, `Savings`, `Collateral_Value`, `Loan_Amount`, `Loan_Term`, `Loan_Purpose`, `Property_Area`, `Education_Level`, `Gender`, `Employer_Category`
- Target: `Loan_Approved` (Yes/No)

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- **Environment:** Jupyter Lab (Anaconda)

## 🔍 Project Workflow

1. **Data Cleaning** – Handled missing values using `SimpleImputer` (mean for numerical columns, most frequent for categorical columns)
2. **Exploratory Data Analysis (EDA)** – Analyzed class balance, income distributions, and outliers via pie charts, bar plots, histograms, and boxplots
3. **Feature Encoding** – Label Encoding for binary/target columns, One-Hot Encoding for nominal categorical columns
4. **Correlation Analysis** – Identified key predictors (`Credit_Score`, `DTI_Ratio`) via a correlation heatmap
5. **Feature Scaling & Train-Test Split** – 80/20 split with `StandardScaler`
6. **Model Training** – Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes
7. **Model Evaluation** – Precision, Recall, F1-score, Accuracy, Confusion Matrix

## 📈 Results

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.786 | 0.620 | 0.693 | 0.805 |
| KNN (k=5) | 0.722 | 0.366 | 0.486 | 0.725 |
| Naive Bayes | 0.750 | 0.465 | 0.574 | 0.755 |

**Key insight:** `Credit_Score` and `DTI_Ratio` emerged as the strongest predictors of loan approval, aligning with real-world lending logic.

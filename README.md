💰 Loan Approval Predictor

A machine learning web app that predicts whether a loan application will be approved, based on applicant details like income, credit score, employment status, and more.

🔗 Live Demo: loan-approval-app.streamlit.app

📋 Overview

This project trains and compares three classification models (Logistic Regression, KNN, and Naive Bayes) on a loan approval dataset, then serves the best-performing model through an interactive Streamlit UI. Users fill in an applicant's financial and personal details, and the app predicts Approved / Not Approved along with a confidence score.

✨ Features
Interactive form for entering applicant details (income, credit score, loan amount, employment status, etc.)
Real-time prediction with confidence percentage
Preprocessing pipeline (imputation, label encoding, one-hot encoding, scaling) matched exactly between training and inference
Deployed and publicly accessible via Streamlit Community Cloud
🛠️ Tech Stack
Python
scikit-learn — model training (Logistic Regression, KNN, Naive Bayes)
pandas / numpy — data processing
Streamlit — web UI and deployment
joblib — model/artifact serialization
📊 Model

The app uses a Logistic Regression model, selected after comparing it against KNN and Naive Bayes on precision, recall, F1 score, and accuracy.

Preprocessing steps applied to the raw data:

Missing value imputation (mean for numeric columns, most frequent for categorical columns)
Label encoding for Education_Level and the target Loan_Approved
One-hot encoding for Employment_Status, Marital_Status, Loan_Purpose, Property_Area, Gender, Employer_Category
Feature scaling with StandardScaler
📁 Project Structure
loan-approval-app/
├── app.py                          # Streamlit UI application
├── save_artifacts.py               # Script to export trained model/encoders/scaler
├── requirements.txt                # Python dependencies
├── loan_model.pkl                  # Trained Logistic Regression model
├── scaler.pkl                      # Fitted StandardScaler
├── label_encoder_education.pkl     # LabelEncoder for Education_Level
├── label_encoder_target.pkl        # LabelEncoder for Loan_Approved
├── onehot_encoder.pkl              # OneHotEncoder for categorical features
├── feature_columns.pkl             # Exact feature column order expected by the model
└── README.md
🚀 Running Locally
Clone the repo:
bash
   git clone https://github.com/yourusername/loan-approval-app.git
   cd loan-approval-app
Install dependencies:
bash
   pip install -r requirements.txt
Run the app:
bash
   streamlit run app.py
Open http://localhost:8501 in your browser.
🧠 Input Fields
Field	Description
Applicant / Coapplicant Income	Monthly or annual income
Age	Applicant's age
Dependents	Number of dependents
Credit Score	Applicant's credit score
Existing Loans	Number of currently active loans
DTI Ratio	Debt-to-income ratio
Savings	Total savings
Collateral Value	Value of offered collateral
Loan Amount / Term	Requested loan amount and repayment term
Employment Status	Salaried / Self-employed / Contract / Unemployed
Marital Status	Single / Married
Education Level	Graduate / Not Graduate
Gender	Male / Female
Employer Category	Private / Government / MNC / Business / Unemployed
Property Area	Urban / Semiurban / Rural
Loan Purpose	Personal / Car / Business / Home / Education
📈 Future Improvements
Add model selection toggle (compare Logistic Regression / KNN / Naive Bayes live)
Add input validation and helpful tooltips
Add SHAP-based explainability for individual predictions
Track prediction history
📄 License

This project is open source and available under the MIT License.



# 📞 Telco Customer Churn Prediction

This is a project focused on predicting customer churn for a telecommunications company. The solution includes a machine learning model built with robust data techniques and deployed as an interactive web application using Streamlit.


<img width="1919" height="968" alt="image" src="https://github.com/user-attachments/assets/d8ab0326-2b8d-49e3-9f24-0b96e48886bb" />


## 🌟 Project Summary

| Component | Technology / Technique | Purpose |
| :--- | :--- | :--- |
| **Model** | **Random Forest Classifier** (`model.pkl`) | Predicts if a customer will churn (Yes/No). |
| **Imbalance Handling** | **SMOTEENN** (`imblearn`) | Corrects the bias in the dataset for better predictive power. |
| **Deployment** | **Streamlit** (`streamlit_app.py`) | Provides a simple web interface for real-time predictions. |
| **Analysis** | **One-Hot Encoding** | Converts categorical features (like Contract Type) into the numerical format required by the model. |

-----
** Quick Start Guide (Deployed App)**
This application is live and publicly accessible. No installation or local execution is required.

1. Access the Application
Click the link below to open the Telco Customer Churn Prediction app in your web browser:

https://churn-prediction-model-tele-co-2sl7xryaufxwnvtsdrpqnz.streamlit.app/

2. Usage
Use the sidebar to input the various features of a new or existing customer (e.g., tenure, contract type, monthly charges).

Click the Predict Churn button.

The main panel will display the prediction result (Likely to Churn or Likely to Stay) and the model's confidence score.

## 📁 Key Files

  * **`streamlit_app.py`**: The main executable script for the live application.
  * **`model (1).pkl`**: The trained and saved Random Forest model.
  * **`Model_Building.ipynb`**: Details the full ML pipeline, including preprocessing, SMOTEENN application, and model training.
  * **`WA_Fn-UseC_-Telco-Customer-Churn (1).csv`**: The raw project dataset.

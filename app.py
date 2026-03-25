import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --------------------------------------------------------------------------------------
# ⚠️ SOLUTION: st.set_page_config() MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide") 
# --------------------------------------------------------------------------------------

# --- 1. Load the Trained Model (Cached for Performance) ---
# The model is loaded once and cached to speed up the app.

MODEL_PATH = 'model.pkl'

@st.cache_resource
def load_model():
    """Loads the trained Random Forest model using pickle."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file '{MODEL_PATH}' not found in the current directory.")
        return None
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model()

# --- 2. Preprocessing Logic ---
# Function to convert user input into the one-hot encoded DataFrame 
# expected by the model (based on the columns in 'tel_churn.csv').

def preprocess_input(input_data):
    """
    Converts raw user input into the one-hot encoded DataFrame
    required by the Random Forest model.
    """
    # 1. Initialize a DataFrame with all expected feature columns set to 0 (False)
    # (Feature columns list remains the same)
    feature_columns = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 
        'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes', 
        'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes', 
        'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 
        'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
        'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
        'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
        'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes', 
        'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes', 
        'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 
        'PaperlessBilling_No', 'PaperlessBilling_Yes', 
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36', 
        'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72'
    ]
    
    # Create the base DataFrame for one customer
    df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # 2. Map Direct Numerical Features
    df['SeniorCitizen'] = 1 if input_data['SeniorCitizen'] == 'Yes' else 0
    df['MonthlyCharges'] = input_data['MonthlyCharges']
    df['TotalCharges'] = input_data['TotalCharges']
    
    # 3. Map Categorical Features (One-Hot Encoding)
    
    # --- START OF CORRECTION ---
    
    # Gender is a special case (Male/Female)
    if input_data['Gender'] == 'Male':
        df['gender_Male'] = 1
    else: # 'Female'
        df['gender_Female'] = 1
        
    # All other Simple Binary Features (Partner, Dependents, PaperlessBilling)
    for col, value in [('Partner', input_data['Partner']),
                       ('Dependents', input_data['Dependents']),
                       ('PaperlessBilling', input_data['PaperlessBilling'])]:
        # Maps 'Yes' to the corresponding '_Yes' column
        if value == 'Yes':
            df[f'{col}_Yes'] = 1
        else:
            # Handles 'No'
            df[f'{col}_No'] = 1 
            
    # --- END OF CORRECTION ---
            
    # PhoneService mapping (No change)
    if input_data['PhoneService'] == 'Yes':
        df['PhoneService_Yes'] = 1
    else:
        df['PhoneService_No'] = 1
        
    # Multi-Category Features (e.g., InternetService, Contract) (No change)
    def map_multi_category(feature_name, user_value):
        if user_value == 'No internet service':
            df[f'{feature_name}_No internet service'] = 1
        else:
            df[f'{feature_name}_{user_value}'] = 1

    map_multi_category('MultipleLines', input_data['MultipleLines'])
    map_multi_category('InternetService', input_data['InternetService'])
    map_multi_category('OnlineSecurity', input_data['OnlineSecurity'])
    map_multi_category('OnlineBackup', input_data['OnlineBackup'])
    map_multi_category('DeviceProtection', input_data['DeviceProtection'])
    map_multi_category('TechSupport', input_data['TechSupport'])
    map_multi_category('StreamingTV', input_data['StreamingTV'])
    map_multi_category('StreamingMovies', input_data['StreamingMovies'])
    
    # Contract and Payment Method are direct maps (No change)
    df[f'Contract_{input_data["Contract"]}'] = 1
    df[f'PaymentMethod_{input_data["PaymentMethod"]}'] = 1

    # 4. Tenure Grouping Logic (No change)
    tenure = input_data['Tenure']
    if 1 <= tenure <= 12:
        df['tenure_group_1 - 12'] = 1
    elif 13 <= tenure <= 24:
        df['tenure_group_13 - 24'] = 1
    elif 25 <= tenure <= 36:
        df['tenure_group_25 - 36'] = 1
    elif 37 <= tenure <= 48:
        df['tenure_group_37 - 48'] = 1
    elif 49 <= tenure <= 60:
        df['tenure_group_49 - 60'] = 1
    elif tenure > 60:
        df['tenure_group_61 - 72'] = 1
        
    return df
# --- 3. Streamlit User Interface ---

st.title("📞 Telco Customer Churn Prediction App")
st.markdown("Use this form to input customer data and predict the likelihood of churn.")

if model is not None:
    # Use st.sidebar for a cleaner main layout
    with st.sidebar:
        st.header("Customer Profile Input")
        
        # --- Collect Input Data ---
        
        # Section 1: Demographics
        st.subheader("Demographics")
        gender = st.radio("Gender", ('Male', 'Female'))
        senior_citizen = st.radio("Senior Citizen", ('No', 'Yes'))
        partner = st.radio("Partner", ('No', 'Yes'))
        dependents = st.radio("Dependents", ('No', 'Yes'))
        
        # Section 2: Account Information
        st.subheader("Account Details")
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ('Month-to-month', 'One year', 'Two year'))
        paperless_billing = st.radio("Paperless Billing", ('No', 'Yes'))
        payment_method = st.selectbox("Payment Method", 
                                      ('Electronic check', 'Mailed check', 
                                       'Bank transfer (automatic)', 'Credit card (automatic)'))
        
        # Section 3: Services
        st.subheader("Services Subscribed")
        phone_service = st.radio("Phone Service", ('No', 'Yes'))
        if phone_service == 'Yes':
            multiple_lines = st.selectbox("Multiple Lines", ('No', 'Yes'))
        else:
            multiple_lines = 'No phone service'
            
        internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
        
        # Helper function for internet service features
        def service_selectbox(label, internet_service_val):
            if internet_service_val == 'No':
                return 'No internet service'
            return st.selectbox(label, ('No', 'Yes'))

        online_security = service_selectbox("Online Security", internet_service)
        online_backup = service_selectbox("Online Backup", internet_service)
        device_protection = service_selectbox("Device Protection", internet_service)
        tech_support = service_selectbox("Tech Support", internet_service)
        streaming_tv = service_selectbox("Streaming TV", internet_service)
        streaming_movies = service_selectbox("Streaming Movies", internet_service)
        
        # Section 4: Charges
        st.subheader("Charges")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        # Total charges should be greater than or equal to (Monthly Charges * tenure), and not less than Monthly Charges.
        total_charges = st.number_input("Total Charges ($)", 
                                        min_value=max(monthly_charges, monthly_charges * (tenure/72)), 
                                        value=monthly_charges * tenure)

    # --- 4. Prediction Logic ---

    # Store all inputs in a dictionary
    raw_input = {
        'SeniorCitizen': senior_citizen, 'MonthlyCharges': monthly_charges, 
        'TotalCharges': total_charges, 'Gender': gender, 'Partner': partner, 
        'Dependents': dependents, 'PhoneService': phone_service, 
        'MultipleLines': multiple_lines, 'InternetService': internet_service, 
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 
        'DeviceProtection': device_protection, 'TechSupport': tech_support, 
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies, 
        'Contract': contract, 'PaperlessBilling': paperless_billing, 
        'PaymentMethod': payment_method, 'Tenure': tenure
    }

    # Prediction button
    if st.button('Predict Churn', type="primary"):
        try:
            # Preprocess the raw input
            final_df = preprocess_input(raw_input)
            
            # Make prediction (0 or 1) and probability
            prediction = model.predict(final_df)[0]
            probability = model.predict_proba(final_df)[0]
            
            # Get the confidence score for the predicted class
            confidence = probability[prediction] * 100
            
            # --- Display Results ---
            st.markdown("---")
            st.header("Prediction Result")
            
            if prediction == 1:
                st.error(f"🚨 **CHURN PREDICTED**")
                st.metric(label="Likelihood to Churn", 
                          value=f"{confidence:.2f}%", 
                          delta=f"{(100 - confidence):.2f}% Likelihood to Stay", 
                          delta_color="inverse")
                st.subheader("Recommendation: High-Risk Customer")
                st.markdown("This customer is highly likely to discontinue service. Consider targeted retention offers.")
            else:
                st.success(f"✅ **CUSTOMER WILL LIKELY STAY**")
                st.metric(label="Likelihood to Stay", 
                          value=f"{confidence:.2f}%", 
                          delta=f"{(100 - confidence):.2f}% Likelihood to Churn", 
                          delta_color="off")
                st.subheader("Recommendation: Low-Risk Customer")
                st.markdown("This customer is expected to continue service. Standard service protocols apply.")
                
            st.markdown("---")
            with st.expander("Show Model Input (for debugging)"):
                st.dataframe(final_df.T)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


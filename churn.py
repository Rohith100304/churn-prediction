import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the pre-trained model
model = load_model('churn prediction knn')

# Load the dataset to get column names and unique values
df = pd.read_csv('churn prediction.csv')

# Streamlit app
st.title('Churn Prediction App')

# User input fields
st.sidebar.header('User Input Features')

# Function to get user input
def user_input_features():
    customerID = "0000-XXXX"  # Placeholder for customerID
    gender = st.selectbox('Gender', df['gender'].unique())
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', df['SeniorCitizen'].unique())
    Partner = st.sidebar.selectbox('Partner', df['Partner'].unique())
    Dependents = st.sidebar.selectbox('Dependents', df['Dependents'].unique())
    tenure = st.sidebar.number_input('Tenure', min_value=0, max_value=100, value=1)
    PhoneService = st.sidebar.selectbox('Phone Service', df['PhoneService'].unique())
    MultipleLines = st.sidebar.selectbox('Multiple Lines', df['MultipleLines'].unique())
    InternetService = st.sidebar.selectbox('Internet Service', df['InternetService'].unique())
    OnlineSecurity = st.sidebar.selectbox('Online Security', df['OnlineSecurity'].unique())
    OnlineBackup = st.sidebar.selectbox('Online Backup', df['OnlineBackup'].unique())
    DeviceProtection = st.sidebar.selectbox('Device Protection', df['DeviceProtection'].unique())
    TechSupport = st.sidebar.selectbox('Tech Support', df['TechSupport'].unique())
    StreamingTV = st.sidebar.selectbox('Streaming TV', df['StreamingTV'].unique())
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', df['StreamingMovies'].unique())
    Contract = st.sidebar.selectbox('Contract', df['Contract'].unique())
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', df['PaperlessBilling'].unique())
    PaymentMethod = st.sidebar.selectbox('Payment Method', df['PaymentMethod'].unique())
    MonthlyCharges = st.sidebar.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=50.0)
    TotalCharges = st.sidebar.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=1000.0)

    data = {
        'customerID': customerID,  # Add customerID as a placeholder
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input features')
st.write(input_df)

# Predict button
if st.sidebar.button('Predict'):
    # Make prediction
    prediction = predict_model(model, data=input_df)
    
    # Display prediction
    st.subheader('Prediction')
    st.write(prediction[['prediction_label', 'prediction_score']])

# Run the app
if __name__ == '__main__':
    st.write('To start, please input your data on the left sidebar and click the Predict button.')

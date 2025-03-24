import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the pre-trained model
model = load_model('churn prediction knn')

# Load the dataset
df = pd.read_csv("churn prediction.csv")

# Streamlit app title
st.title("Customer Churn Prediction")

# **Placeholders to prevent content shifting**
dataset_placeholder = st.empty()
input_form_placeholder = st.empty()
prediction_placeholder = st.empty()

# **Sidebar Controls**
st.sidebar.header("Options")

# **View Dataset Button**
if st.sidebar.button("View Dataset"):
    with dataset_placeholder.container():
        st.write("### Churn Prediction Dataset")
        st.dataframe(df)

# **Download Dataset Button**
st.sidebar.download_button(
    label="Download Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="churn_prediction.csv",
    mime="text/csv"
)

# **Download Model Button**
st.sidebar.download_button(
    label="Download Model",
    data=open("churn prediction knn.pkl", "rb"),
    file_name="churn_prediction_knn.pkl",
    mime="application/octet-stream"
)

# **ABC Button - Show Input Form**
if st.sidebar.button("ABC"):
    with input_form_placeholder.container():
        st.write("### Enter Customer Details")

        # **Customer Input Fields**
        gender = st.selectbox('Gender', df['gender'].unique())
        SeniorCitizen = st.selectbox('Senior Citizen', df['SeniorCitizen'].unique())
        Partner = st.selectbox('Partner', df['Partner'].unique())
        Dependents = st.selectbox('Dependents', df['Dependents'].unique())
        tenure = st.number_input('Tenure', min_value=0, max_value=100, value=1)
        PhoneService = st.selectbox('Phone Service', df['PhoneService'].unique())
        MultipleLines = st.selectbox('Multiple Lines', df['MultipleLines'].unique())
        InternetService = st.selectbox('Internet Service', df['InternetService'].unique())
        OnlineSecurity = st.selectbox('Online Security', df['OnlineSecurity'].unique())
        OnlineBackup = st.selectbox('Online Backup', df['OnlineBackup'].unique())
        DeviceProtection = st.selectbox('Device Protection', df['DeviceProtection'].unique())
        TechSupport = st.selectbox('Tech Support', df['TechSupport'].unique())
        StreamingTV = st.selectbox('Streaming TV', df['StreamingTV'].unique())
        StreamingMovies = st.selectbox('Streaming Movies', df['StreamingMovies'].unique())
        Contract = st.selectbox('Contract', df['Contract'].unique())
        PaperlessBilling = st.selectbox('Paperless Billing', df['PaperlessBilling'].unique())
        PaymentMethod = st.selectbox('Payment Method', df['PaymentMethod'].unique())
        MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=50.0)
        TotalCharges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=1000.0)

        # **Prepare DataFrame for Prediction**
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'InternetService': [InternetService],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV],
            'StreamingMovies': [StreamingMovies],
            'Contract': [Contract],
            'PaperlessBilling': [PaperlessBilling],
            'PaymentMethod': [PaymentMethod],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]
        })

        # **Show User Input**
        st.subheader("User Input Data")
        st.write(input_data)

        # **Predict Button**
        if st.button("Predict"):
            prediction = predict_model(model, data=input_data)

            # **Display Prediction Result**
            with prediction_placeholder.container():
                st.subheader("Prediction Result")
                st.write(f"**Churn Prediction:** {prediction['prediction_label'].values[0]}")
                st.write(f"**Confidence Score:** {prediction['prediction_score'].values[0]:.2f}")

# **Main Page Content**
st.write("### Welcome to the Churn Prediction App!")
st.write(
    "Use the sidebar to **view the dataset, download files, and access customer details**. "
    "Click **ABC** to enter customer details and **Predict** to see churn probability."
)

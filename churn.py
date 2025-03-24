import streamlit as st
import pandas as pd
import pickle
from pycaret.classification import load_model, predict_model

# Load the pre-trained model
model = load_model('churn prediction knn')

# Save the model as a pickle file (if not already saved)
model_filename = "churn_prediction_knn.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(model, model_file)

# Load the dataset
df = pd.read_csv("churn prediction.csv")

# Streamlit app title
st.title("Customer Churn Prediction")

# **Placeholders to control dynamic content**
dataset_placeholder = st.empty()
input_form_placeholder = st.empty()
prediction_placeholder = st.empty()

# Sidebar controls
st.sidebar.header("Options")

# 1st Button - View Dataset (Controlled with st.empty())
if st.sidebar.button("View Dataset"):
    dataset_placeholder.write("### Churn Prediction Dataset")
    dataset_placeholder.dataframe(df)

# 2nd Button - Download Dataset
st.sidebar.download_button(
    label="Download Dataset",
    data=df.to_csv(index=False).encode(),
    file_name="churn_prediction.csv",
    mime="text/csv"
)

# 3rd Button - Download Model
with open(model_filename, "rb") as model_file:
    st.sidebar.download_button(
        label="Download Model",
        data=model_file,
        file_name="churn_prediction_knn.pkl",
        mime="application/octet-stream"
    )

# **ABC Button (Displays User Input Form)**
if st.sidebar.button("Predict Data"):
    def user_input_features():
    customerID = "0000-XXXX"  # Placeholder for customerID
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
if st.button('Predict'):
    # Make prediction
    prediction = predict_model(model, data=input_df)
    
    # Display prediction
    st.subheader('Prediction')
    st.write(prediction[['prediction_label', 'prediction_score']])
# **Main Page Content (Static, Unaffected by Buttons)**
#st.write("### Welcome to the Churn Prediction App!")
#st.write(
 #   "Use the sidebar to **view the dataset, download files, and access customer details**. "
#    "Click **ABC** to enter customer details and **Predict** to see churn probability."
#)

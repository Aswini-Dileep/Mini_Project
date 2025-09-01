import streamlit as st
import pandas as pd
import numpy as np
from os import path
import pickle

st.title("Customer Churn Predictor")

gender = st.radio("Select Gender", ["Male", "Female"], index=None)
SeniorCitizen = st.radio("Please choose senior citizen or not", ["Yes", "No"], index=None)
tenure = st.number_input("Please choose tenure", min_value=0.0, max_value=72.0,value=None, placeholder="Please enter a valid number between 0.0 and 72.0")
PhoneService = st.radio("Whether customer has a phone service", ["Yes", "No"], index=None)
MultipleLines = st.radio("Whether customer has multiple Phone service", ["Yes", "No"], index=None)
InternetService = st.radio("Please choose type of internet service", ["No", "DSL", "Fiber optic"], index=None)
OnlineSecurity = st.radio("Whether customer has a online security", ["Yes", "No"], index=None)
OnlineBackup = st.radio("Whether customer has online backup", ["Yes", "No"], index=None)
DeviceProtection = st.radio("Whether customer has device protection", ["Yes", "No"], index=None)
TechSupport = st.radio("Whether customer has tech support", ["Yes", "No"], index=None)
StreamingTV = st.radio("Whether customer has streaming TV", ["Yes", "No"], index=None)
StreamingMovies = st.radio("Whether customer has Streaming Movies", ["Yes", "No"], index=None)
Contract = st.radio("Please choose contract type", ["One year", "Month-to-month", "Two year"], index=None)
PaperlessBilling = st.radio("Whether customer use paperless billing", ["Yes", "No"], index=None)
PaymentMethod = st.radio("please choose the payment method", ["Mailed check", "Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"], index=None)
MonthlyCharges = st.number_input("Please choose monthly charges", min_value=18.7, max_value=118.6,value=None,
                                 placeholder="Please enter a valid number between 18.7 and 118.6")
TotalCharges = st.number_input("Please choose total charges", min_value=18.8, max_value=8547.15,value=None,
                               placeholder="Please enter a valid number between 18.8 and 8547.15")

#prepare the dataframe for prediction
df_user_input = pd.DataFrame([[gender, SeniorCitizen, tenure, PhoneService, MultipleLines,
                               InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                               TechSupport, StreamingTV, StreamingMovies, Contract,
                               PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]],
                             columns=['Gender', 'SeniorCitizen', 'Tenure', 'PhoneService', 'MultipleLines',
                                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                      'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                      'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
                                      ])
st.write(df_user_input)

# using the .pkl file, creating an ML model named 'churn_predictor'
model_path = path.join("Model", "RandomForest_pipeline.pkl")
with open (model_path, 'rb') as file:
    churn_predictor = pickle.load(file)
dict_churn = {0:"No", 1:"Yes"}

if st.button("Predict Churn"):
    if((gender==None) or (SeniorCitizen==None) or (tenure==None) or
            (PhoneService==None) or (MultipleLines==None) or (InternetService==None) or
            (OnlineSecurity==None) or (OnlineBackup==None) or (DeviceProtection==None) or
            (TechSupport==None) or (StreamingTV==None) or (StreamingMovies==None) or
            (Contract==None) or (PaperlessBilling==None) or (PaymentMethod==None) or
            (MonthlyCharges==None) or (TotalCharges==None)):
         st.write("Please fill all values")
    else:
         predicted_churn = churn_predictor.predict(df_user_input)

         st.write("The Churn is ", dict_churn[predicted_churn[0]])

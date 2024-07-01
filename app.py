import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
model = load_model('your_model.h5')

# Load the scaler
scaler = joblib.load('your_scaler.pkl')

# Define the input features
feature_names = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
                "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]

def preprocess_input(data):
    # Convert data to DataFrame
    input_df = pd.DataFrame(data, index=[0])
    # Scale the data
    scaled_data = scaler.transform(input_df)
    return scaled_data

# Streamlit app code
st.title('Heart Failure Prediction')

user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f'Enter {feature}', key=feature)

if st.button('predict'):
    input_data = preprocess_input(user_input)
    prediction = model.predict(input_data)
    output = 'Yes' if prediction[0][0] > 0.5 else 'No'
    st.write(f'Heart Failure Prediction: {output}')
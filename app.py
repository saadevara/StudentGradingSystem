import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression

# Load the trained model and feature names
model = joblib.load('model.joblib')
feature_names = joblib.load('feature_names.joblib')

# Function to preprocess input data
def preprocess_data(data):
    # One-hot encode 'reason' column
    data = pd.get_dummies(data, columns=['reason'], drop_first=True)
    # Ensure all columns are present, add missing columns with zero values
    missing_cols = set(feature_names) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    # Reorder columns to match the order during training
    data = data[feature_names]
    return data

# Streamlit app
def main():
    st.title("Student Grade Prediction App")

    # Collect user input
    st.sidebar.header("User Input")
    failures = st.sidebar.number_input("Number of Failures", min_value=0, max_value=3, value=0)
    Medu = st.sidebar.number_input("Mother's Education Level", min_value=0, max_value=4, value=1)
    higher = st.sidebar.checkbox("Wants to Pursue Higher Education", value=True)
    age = st.sidebar.number_input("Age", min_value=15, max_value=22, value=18)
    Fedu = st.sidebar.number_input("Father's Education Level", min_value=0, max_value=4, value=4)
    goout = st.sidebar.number_input("Going Out with Friends (1-5)", min_value=1, max_value=5, value=4)
    romantic = st.sidebar.checkbox("In a Romantic Relationship", value=False)
    reason = st.sidebar.selectbox("Reason for Choosing this School", options=["home", "reputation", "course", "other"], index=0)

    # Create a DataFrame from user input
    user_data = pd.DataFrame({
        'failures': [failures],
        'Medu': [Medu],
        'higher': [1 if higher else 0],
        'age': [age],
        'Fedu': [Fedu],
        'goout': [goout],
        'romantic': [1 if romantic else 0],
        'reason': [reason]
    })

    # Preprocess the user data
    user_data = preprocess_data(user_data)

    # Display the user input
    st.subheader("User Input:")
    st.write(user_data)

    # Make predictions
    prediction = model.predict(user_data)

    # Display the prediction
    st.subheader("Predicted Grade (G3):")
    st.write(prediction[0])

if __name__ == "__main__":
    main()

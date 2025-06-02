import streamlit as st
import pandas as pd
import numpy as np
import pickle
from prediction_helper import prediction_helper
# Load your model (uncomment when model is available)
# with open('health_premium_model.pkl', 'rb') as f:
#     model = pickle.load(f)

st.set_page_config(page_title="Health Premium Predictor", layout="centered")

st.title("🩺 Health Insurance Premium Prediction App")

st.markdown("Fill in the details below to estimate your annual health insurance premium.")

# Layout: Use columns for better alignment
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'], key='gender')
    region = st.selectbox("Region", ['Northwest', 'Southeast', 'Northeast', 'Southwest'], key='region')
    marital_status = st.selectbox("Marital Status", ['Unmarried', 'Married'], key='marital')
    bmi_category = st.selectbox("BMI Category", ['Normal', 'Obesity', 'Overweight', 'Underweight'], key='bmi')
    smoking_status = st.selectbox("Smoking Status", [
        'No Smoking', 'Regular', 'Occasional', 'Smoking=0', 'Does Not Smoke', 'Not Smoking'
    ], key='smoke')
    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)



with col2:
    employment_status = st.selectbox("Employment Status", ['Salaried', 'Self-Employed', 'Freelancer', 'None'], key='job')

    medical_history = st.selectbox("Medical History", [
        'Diabetes', 'High blood pressure', 'No Disease', 'Diabetes & High blood pressure',
        'Thyroid', 'Heart disease', 'High blood pressure & Heart disease',
        'Diabetes & Thyroid', 'Diabetes & Heart disease'
    ], key='history')
    insurance_plan = st.selectbox("Insurance Plan", ['Bronze', 'Silver', 'Gold'], key='plan')
    genetic_risk = st.number_input("Genetic Risk", min_value=0, max_value=100, value=30, step=1)
    income_lakhs = st.number_input("Income (in Lakhs)", min_value=0.0, step=0.5)

    age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)

    input_dir  = {
        'Gender': gender,
        'Region': region,
        'Marital_status': marital_status,
        'BMI_Category': bmi_category,
        'Smoking_Status': smoking_status,
        'Employment_Status': 'None' if employment_status == 'None' else employment_status,
        'Medical History': medical_history,
        'Insurance_Plan': insurance_plan,
        'Genetic_Risk': genetic_risk,
        'Income_Lakhs': income_lakhs,
        'Age': age ,

        'Num dependents' : num_dependents,

    }


# Predict Button
if st.button("💡 Predict Premium", key="predict_button_main"):
    input_data = pd.DataFrame([{
        'Gender': gender,
        'Region': region,
        'Marital_status': marital_status,
        'BMI_Category': bmi_category,
        'Smoking_Status': smoking_status,
        'Employment_Status': 'None' if employment_status == 'None' else employment_status,
        'Medical History': medical_history,
        'Insurance_Plan': insurance_plan
    }])


    predicted = prediction_helper(input_dir)

    st.success(f'predicted premium: {predicted}')


    # st.subheader("🔍 Input Summary")
    # st.write(input_data)

    # Predict using the model (replace this with actual prediction)
    # prediction = model.predict(input_data)[0]
    # prediction = np.random.randint(15000, 75000)

    # st.success(f"💰 Estimated Annual Premium: ₹{prediction:,.2f}")
    # st.info("Note: This is a simulated result. Actual premium may vary.")

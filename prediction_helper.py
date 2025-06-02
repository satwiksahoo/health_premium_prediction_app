from joblib import load
import pandas as pd

model_rest  = load('artifacts/model_rest.joblib')
model_young = load('artifacts/model_young.joblib')

scaler_rest  = load('artifacts/scaler_rest.joblib')
scaler_young = load('artifacts/scaler_young.joblib')


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)

    max_score = 14
    min_score = 0
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score


def process_inputs(input_dict):
    base_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=base_columns, index=[0])

    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            df[f'region_{value}'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            df[f'bmi_category_{value}'] = 1
        elif key == 'Smoking Status':
            df[f'smoking_status_{value}'] = 1
        elif key == 'Employment Status':
            df[f'employment_status_{value}'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    scaled_df = handle_scaling(input_dict['Age'], df )
    return scaled_df



def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    scaler = scaler_object['scaler']
    cols_to_scale = scaler_object['cols_to_scale']



    # Apply scaling
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])


    return df
def prediction_helper(input_dict):
    age = input_dict['Age']
    input_df = process_inputs(input_dict)

    if age <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    # print(prediction)
    return int(prediction)

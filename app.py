import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title='Disease Prediction System',
    layout='wide',
    page_icon='🧑‍⚕️'
)

# Load Models & Scalers
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

diabetes_model = load_model("C:\\2k25\\AICTE Diseases Breakout\\diabetes_model (3).pkl")
heart_disease_model = load_model("C:\\2k25\\AICTE Diseases Breakout\\heart_disease_model.pkl")
diabetes_scaler = load_model("C:\\2k25\\AICTE Diseases Breakout\\diabetes_scaler (3).pkl")  # Ensure scaler is saved
heart_scaler = load_model("C:\\2k25\\AICTE Diseases Breakout\\heart_scaler.pkl")  # Save and load accordingly

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction"],
        icons=["activity", "heart"],
        default_index=0
    )

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("🔬 Diabetes Prediction")
    st.markdown("Fill in the details below to predict diabetes.")
    
    # User Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
    with col2:
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, step=1)
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
    with col3:
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, step=0.1)
    
    # Prediction
    if st.button("Predict Diabetes", use_container_width=True):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        input_data_scaled = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_data_scaled)[0]
        result = "✅ The person is NOT diabetic." if prediction == 0 else "⚠️ The person is DIABETIC."
        st.success(result)

# Define Expected Feature Names
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
st.title("❤️ Heart Disease Prediction")
st.markdown("Fill in the details below to predict heart disease.")

# User Inputs
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, step=1, key="age_input")
    sex = st.radio("Sex", ["Male", "Female"], key="sex_input")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x], 
                      key="cp_input")
    resting_bp = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=200, step=1, key="bp_input")

with col2:
    cholesterol = st.number_input("Cholesterol Level (chol)", min_value=100, max_value=600, step=1, key="chol_input")
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], key="fbs_input")
    restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], 
                           format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x], 
                           key="restecg_input")
    max_heart_rate = st.number_input("Max Heart Rate (thalach)", min_value=50, max_value=250, step=1, key="hr_input")

with col3:
    exercise_angina = st.radio("Exercise-Induced Angina (exang)", [0, 1], key="angina_input")
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1, key="oldpeak_input")
    slope = st.selectbox("ST Slope", options=[0, 1, 2], 
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x], 
                         key="slope_input")
    ca = st.slider("Number of Major Vessels Colored (ca)", 0, 4, 0, key="ca_input")
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], 
                        format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x], 
                        key="thal_input")

# Convert categorical inputs
sex = 1 if sex == "Male" else 0  # Convert Male/Female to 1/0

# Prediction
if st.button("Predict Heart Disease", use_container_width=True):
    # Prepare input data as DataFrame (Ensuring correct feature order)
    input_data_dict = {
        "age": [age], "sex": [sex], "cp": [cp], "trestbps": [resting_bp], 
        "chol": [cholesterol], "fbs": [fasting_bs], "restecg": [restecg], 
        "thalach": [max_heart_rate], "exang": [exercise_angina], "oldpeak": [oldpeak], 
        "slope": [slope], "ca": [ca], "thal": [thal]
    }
    
    input_data_df = pd.DataFrame(input_data_dict)

    # Check if All Features Exist Before Scaling
    missing_features = set(feature_names) - set(input_data_df.columns)
    if missing_features:
        st.error(f"Missing Features: {missing_features}")
    else:
        input_data_scaled = heart_scaler.transform(input_data_df)  # Apply same transformation
        prediction = heart_disease_model.predict(input_data_scaled)[0]

        result = "✅ The person is NOT at risk of heart disease." if prediction == 0 else "⚠️ The person is at risk of HEART DISEASE."
        st.success(result)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by Vandan")

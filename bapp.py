import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model using pickle
with open('XGBoost_model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a Streamlit web app
st.title("Credit Card Default Prediction Dashboard")

# Navigation bar
st.markdown("""
<nav>
    <ul>
        <li><a href="#basic-info">Basic Information</a></li>
        <li><a href="#repayment-status">Repayment Status</a></li>
        <li><a href="#bill-statements">Bill Statements</a></li>
        <li><a href="#previous-payments">Previous Payments</a></li>
    </ul>
</nav>
""", unsafe_allow_html=True)

st.markdown("<h2 id='basic-info'>Basic Information</h2>", unsafe_allow_html=True)

# Input fields for basic information
limit_bal = st.slider("Credit Limit (NT dollars)", 0, 1000000, 50000)
sex = st.radio("Gender", ["Male", "Female"])
education = st.radio("Education Level", ["Graduate School", "University", "High School", "Others"])
marriage = st.radio("Marital Status", ["Married", "Single", "Others"])
age = st.slider("Age (years)", 20, 80, 30)

st.markdown("<h2 id='repayment-status'>Repayment Status (2005)</h2>", unsafe_allow_html=True)

# Input fields for repayment status
pay_status_sept = st.slider("Repayment Status - September", -2, 8, 0)
pay_status_aug = st.slider("Repayment Status - August", -2, 8, 0)
pay_status_jul = st.slider("Repayment Status - July", -2, 8, 0)
pay_status_jun = st.slider("Repayment Status - June", -2, 8, 0)
pay_status_may = st.slider("Repayment Status - May", -2, 8, 0)
pay_status_apr = st.slider("Repayment Status - April", -2, 8, 0)

st.markdown("<h2 id='bill-statements'>Bill Statements (NT dollars)</h2>", unsafe_allow_html=True)

# Input fields for bill statements
bill_amt_sept = st.slider("Bill Statement - September", 0, 1000000, 5000)
bill_amt_aug = st.slider("Bill Statement - August", 0, 1000000, 5000)
bill_amt_jul = st.slider("Bill Statement - July", 0, 1000000, 5000)
bill_amt_jun = st.slider("Bill Statement - June", 0, 1000000, 5000)
bill_amt_may = st.slider("Bill Statement - May", 0, 1000000, 5000)
bill_amt_apr = st.slider("Bill Statement - April", 0, 1000000, 5000)

st.markdown("<h2 id='previous-payments'>Previous Payments (NT dollars)</h2>", unsafe_allow_html=True)

# Input fields for previous payments
pay_amt_sept = st.slider("Previous Payment - September", 0, 100000, 500)
pay_amt_aug = st.slider("Previous Payment - August", 0, 100000, 500)
pay_amt_jul = st.slider("Previous Payment - July", 0, 100000, 500)
pay_amt_jun = st.slider("Previous Payment - June", 0, 100000, 500)
pay_amt_may = st.slider("Previous Payment - May", 0, 100000, 500)
pay_amt_apr = st.slider("Previous Payment - April", 0, 100000, 500)

# Define mappings for education and marriage
education_mapping = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}

marriage_mapping = {
    "Married": 1,
    "Single": 2,
    "Others": 3
}

# Create a DataFrame with user input data
user_input_data = pd.DataFrame({
    "LIMIT_BAL": [limit_bal],
    "SEX": [1 if sex == "Male" else 2],  # Map 'Male' to 1 and 'Female' to 2
    "EDUCATION": [education_mapping[education]],
    "MARRIAGE": [marriage_mapping[marriage]],
    "AGE": [age],
    "PAY_0": [pay_status_sept],
    "PAY_2": [pay_status_aug],
    "PAY_3": [pay_status_jul],
    "PAY_4": [pay_status_jun],
    "PAY_5": [pay_status_may],
    "PAY_6": [pay_status_apr],
    "BILL_AMT1": [bill_amt_sept],
    "BILL_AMT2": [bill_amt_aug],
    "BILL_AMT3": [bill_amt_jul],
    "BILL_AMT4": [bill_amt_jun],
    "BILL_AMT5": [bill_amt_may],
    "BILL_AMT6": [bill_amt_apr],
    "PAY_AMT1": [pay_amt_sept],
    "PAY_AMT2": [pay_amt_aug],
    "PAY_AMT3": [pay_amt_jul],
    "PAY_AMT4": [pay_amt_jun],
    "PAY_AMT5": [pay_amt_may],
    "PAY_AMT6": [pay_amt_apr]
})

# Predict button
if st.button("Predict"):
    # Make predictions using the loaded model
    predicted_default = model.predict(user_input_data)

    # Display the prediction result
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Prediction Result")
    if predicted_default[0] == 1:
        st.write("The model predicts that the client may default on their credit card payment.")
    else:
        st.write("The model predicts that the client is unlikely to default on their credit card payment.")
    st.markdown("<hr>", unsafe_allow_html=True)

# Additional information
st.subheader("Model Information")
st.write("This dashboard uses an XGBoost model to predict the likelihood of a client defaulting on their credit card payment based on their financial and demographic information.")
st.write("Adjust the sliders and options to see how changes in the inputs affect the prediction.")

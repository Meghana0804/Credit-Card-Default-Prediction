import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model using pickle
with open('XGBoost_model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a Streamlit web app
st.title("Credit Card Default Prediction Dashboard")

# Add input fields for user input
st.sidebar.header("User Input Features")

# Group input fields into collapsible sections
with st.sidebar.expander("Basic Information"):
    limit_bal = st.slider("LIMIT_BAL (Amount of Credit in NT dollars)", 0, 1000000, 50000)
    sex = st.radio("SEX (Gender)", ["Male", "Female"])
    education = st.radio("EDUCATION (Education Level)", ["Graduate School", "University", "High School", "Others"])
    marriage = st.radio("MARRIAGE (Marital Status)", ["Married", "Single", "Others"])
    age = st.slider("AGE (Age in years)", 20, 80, 30)

with st.sidebar.expander("Repayment Status (2005)"):
    pay_status_sept = st.slider("PAY_0 (September)", -2, 8, 0)
    pay_status_aug = st.slider("PAY_2 (August)", -2, 8, 0)
    pay_status_jul = st.slider("PAY_3 (July)", -2, 8, 0)
    pay_status_jun = st.slider("PAY_4 (June)", -2, 8, 0)
    pay_status_may = st.slider("PAY_5 (May)", -2, 8, 0)
    pay_status_apr = st.slider("PAY_6 (April)", -2, 8, 0)

with st.sidebar.expander("Bill Statements (NT dollars)"):
    bill_amt_sept = st.slider("BILL_AMT1 (September)", 0, 1000000, 5000)
    bill_amt_aug = st.slider("BILL_AMT2 (August)", 0, 1000000, 5000)
    bill_amt_jul = st.slider("BILL_AMT3 (July)", 0, 1000000, 5000)
    bill_amt_jun = st.slider("BILL_AMT4 (June)", 0, 1000000, 5000)
    bill_amt_may = st.slider("BILL_AMT5 (May)", 0, 1000000, 5000)
    bill_amt_apr = st.slider("BILL_AMT6 (April)", 0, 1000000, 5000)

with st.sidebar.expander("Previous Payments (NT dollars)"):
    pay_amt_sept = st.slider("PAY_AMT1 (September)", 0, 100000, 500)
    pay_amt_aug = st.slider("PAY_AMT2 (August)", 0, 100000, 500)
    pay_amt_jul = st.slider("PAY_AMT3 (July)", 0, 100000, 500)
    pay_amt_jun = st.slider("PAY_AMT4 (June)", 0, 100000, 500)
    pay_amt_may = st.slider("PAY_AMT5 (May)", 0, 100000, 500)
    pay_amt_apr = st.slider("PAY_AMT6 (April)", 0, 100000, 500)

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
    "SEX": [1 if sex == "Male" else 2],
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
if st.sidebar.button("Predict"):
    # Make predictions using the loaded model
    predicted_default = model.predict(user_input_data)

    # Display the prediction result
    st.subheader("Prediction Result")
    if predicted_default[0] == 1:
        st.write("The model predicts that the client may default on their credit card payment.")
    else:
        st.write("The model predicts that the client is unlikely to default on their credit card payment.")

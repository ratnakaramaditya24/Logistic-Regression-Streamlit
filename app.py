# app.py

import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('logistic_model.pkl')

# Title
st.title("Titanic Survival Prediction App")

# Sidebar input fields
st.sidebar.header("Passenger Input Features")

# Collect user input
Pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.sidebar.selectbox("Sex", ["male", "female"])
Age = st.sidebar.slider("Age", 0, 100, 30)
SibSp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.sidebar.slider("Parents/Children Aboard", 0, 10, 0)
Fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 50.0)
Embarked = st.sidebar.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Convert categorical to numerical (match training encoding)
Sex_male = 1 if Sex == "male" else 0
Embarked_Q = 1 if Embarked == "Q" else 0
Embarked_S = 1 if Embarked == "S" else 0

# Create feature array (match model input order!)
features = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"Passenger is likely to SURVIVE with probability {probability:.2f}")
    else:
        st.error(f"Passenger is likely to NOT survive with probability {1 - probability:.2f}")

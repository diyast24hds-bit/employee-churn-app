import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Employee Churn Predictor", layout="wide")

st.title("📊 Employee Churn Prediction System")

# ---------------- INPUT SECTION ----------------
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.number_input("Monthly Income", 1000, 80000, 5000)
satisfaction = st.sidebar.slider("Job Satisfaction (1-5)", 1, 5, 2)
years = st.sidebar.slider("Years at Company", 0, 40, 5)
overtime = st.sidebar.selectbox("Overtime", ["No", "Yes"])

# Encode Overtime EXACTLY like training
overtime = 1 if overtime == "Yes" else 0

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict"):

    # ⚠️ ORDER MUST MATCH TRAINING
    input_data = np.array([[age, income, satisfaction, years, overtime]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # ---------------- OUTPUT ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Result")

        if prediction == 1:
            st.error(f"⚠️ High Risk of Attrition ({prob:.2f})")
        else:
            st.success(f"✅ Low Risk of Attrition ({prob:.2f})")

        st.write(f"Probability of Leaving: {prob:.2f}")

    # ---------------- CHART ----------------
    with col2:
        st.subheader("Prediction Confidence")

        fig, ax = plt.subplots()
        ax.bar(["Stay", "Leave"], [1 - prob, prob])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

    # ---------------- INSIGHTS ----------------
    st.subheader("🧠 Basic Insights")

    if overtime == 1:
        st.write("- Overtime increases attrition risk")
    if satisfaction <= 2:
        st.write("- Low job satisfaction detected")
    if years < 2:
        st.write("- New employees tend to leave more")

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("Employee Churn Prediction | Internship Project")
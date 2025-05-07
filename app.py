
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(layout="wide", page_title="Farmer Credit Assessment")

@st.cache_resource
def load_model():
    with open("models/credit_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🧑‍🌾 Farmer Credit Assessment Dashboard")

# Sidebar inputs
st.sidebar.header("Enter Farmer Data")
avg_yield = st.sidebar.slider("Average Yield (tons/hectare)", 1.0, 10.0, 3.5)
loan_repayment = st.sidebar.slider("Past Loan Repayment (%)", 0, 100, 85)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)

input_df = pd.DataFrame({
    "avg_yield": [avg_yield],
    "loan_repayment": [loan_repayment],
    "credit_score": [credit_score]
})

if st.sidebar.button("Predict Creditworthiness"):
    prediction = model.predict(input_df)[0]
    pred_text = "✅ Eligible for Loan" if prediction == 1 else "❌ Not Eligible"
    st.sidebar.success(pred_text)

# Dummy dashboard visuals
col1, col2 = st.columns(2)

with col1:
    st.subheader("Farmers by Credit Score Bracket")
    df = pd.DataFrame({
        "Credit Bracket": ["300-500", "501-650", "651-750", "751-850"],
        "Farmers": [50, 120, 80, 30]
    })
    fig = px.bar(df, x="Credit Bracket", y="Farmers", color="Credit Bracket")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Loan Repayment Rate vs Eligibility")
    df = pd.DataFrame({
        "Repayment Rate": [60, 70, 80, 90, 100],
        "Eligible": [20, 40, 60, 80, 100]
    })
    fig2 = px.line(df, x="Repayment Rate", y="Eligible", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

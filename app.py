
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

# Load ML model
model = joblib.load("models/credit_model.pkl")

# Mock Data
data = {
    "name": ["Michael Johnson", "Jane Doe", "Ahmed Musa"],
    "state": ["Iowa", "Texas", "Kano"],
    "crop_type": ["Corn", "Wheat", "Rice"],
    "credit_score": [720, 590, 660],
    "loan_repayment": [92, 75, 85],
    "avg_yield": [3.2, 2.5, 2.9],
    "target_yield": [3.0, 2.8, 3.0],
    "profile_picture": [
        "https://via.placeholder.com/150",
        "https://via.placeholder.com/150",
        "https://via.placeholder.com/150",
    ]
}
df = pd.DataFrame(data)

# Streamlit layout
st.set_page_config(layout="wide")
st.title("Agro Lender Credit Dashboard")

# Sidebar Selection
farmer_names = df["name"].tolist()
selected_farmer = st.sidebar.selectbox("Select Farmer", farmer_names)
selected_data = df[df["name"] == selected_farmer].iloc[0]

# Layout Sections (matching the visual prototype)
col1, col2, col3 = st.columns([2, 3, 4])

# Farmer Profile (Left Column)
with col1:
    st.image(selected_data["profile_picture"], width=150)
    st.subheader(f"Farmer: {selected_data['name']}")
    st.text(f"State: {selected_data['state']}")
    st.text(f"Crop: {selected_data['crop_type']}")

# Credit Score & Loan Eligibility (Middle Column)
with col2:
    st.subheader("Credit Score")

    # Credit Score Gauge (using Plotly)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=selected_data["credit_score"],
        title={'text': "Credit Score", 'font': {'size': 20}},
        delta={'reference': 650, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={'axis': {'range': [None, 800]}, 'bar': {'color': "lightblue"}, 'steps': [
            {'range': [0, 650], 'color': "red"},
            {'range': [650, 800], 'color': "green"}]
        }))
    st.plotly_chart(fig)

    # ML prediction
    features = [[selected_data['avg_yield'], selected_data['loan_repayment'], selected_data['credit_score']]]
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("ML Model: Eligible for Loan")
    else:
        st.error("ML Model: Not Eligible")

# Repayment & Yield (Right Column)
with col3:
    st.subheader("Repayment & Yield")
    st.metric("Loan Repayment", f"{selected_data['loan_repayment']}%")
    fig2 = go.Figure(data=[
        go.Bar(name='Actual Yield', x=["Farm"], y=[selected_data["avg_yield"]]),
        go.Bar(name='Target Yield', x=["Farm"], y=[selected_data["target_yield"]])
    ])
    fig2.update_layout(title="Farm Yield vs Target", barmode='group')
    st.plotly_chart(fig2)

# Sidebar - Loan Simulation
st.sidebar.header("Loan Simulation")
loan_amount = st.sidebar.slider("Loan Amount ($)", 1000, 20000, 5000)
impact = loan_amount / 200
st.sidebar.write(f"Estimated Credit Score Impact: {int(impact)} points")

simulated_score = selected_data["credit_score"] + int(impact)
st.sidebar.write(f"Simulated Credit Score: {simulated_score}")

simulated_features = [[selected_data['avg_yield'], selected_data['loan_repayment'], simulated_score]]
simulated_prediction = model.predict(simulated_features)[0]

if simulated_prediction == 1:
    st.sidebar.success("Loan Eligibility (Simulated): YES")
else:
    st.sidebar.error("Loan Eligibility (Simulated): NO")

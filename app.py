import streamlit as st
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier

# Configure the page
st.set_page_config(
    page_title="Agro Lender Dashboard",
    page_icon="🌾",
    layout="wide"
)

# --------------------------------------------------------------------
# Load mock farmer credit data
@st.cache_data
def load_farmer_data():
    return pd.DataFrame({
        "Farmer ID": ["F001", "F002", "F003", "F004", "F005", "F006"],
        "Location": ["North", "South", "East", "West", "North", "East"],
        "Avg Yield (t/ha)": [3.5, 2.8, 3.1, 1.9, 4.0, 2.2],
        "Credit Score": [710, 620, 680, 540, 760, 600],
        "Loan Repayment %": [92, 78, 85, 63, 97, 74],
        "Year": [2021, 2021, 2021, 2021, 2021, 2021],
        "Eligible": [1, 0, 1, 0, 1, 0]
    })

farmer_df = load_farmer_data()

# --------------------------------------------------------------------
# ML Model (trained on-the-fly)
@st.cache_resource
def train_model(df):
    X = df[["Avg Yield (t/ha)", "Credit Score", "Loan Repayment %"]]
    y = df["Eligible"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model(farmer_df)

# --------------------------------------------------------------------
# Page Content
st.title("🌾 Agro Creditworthiness Dashboard")

st.markdown("""
View farmer credit data interactively. You can filter by region, adjust input values, and predict loan eligibility on-the-fly.
""")

# Sidebar filters
st.sidebar.header("🧮 Predict Farmer Eligibility")

with st.sidebar:
    avg_yield = st.slider("Average Yield (t/ha)", 1.0, 6.0, 3.0)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    loan_repayment = st.slider("Loan Repayment %", 0, 100, 80)

    user_input = pd.DataFrame([{
        "Avg Yield (t/ha)": avg_yield,
        "Credit Score": credit_score,
        "Loan Repayment %": loan_repayment
    }])

    if st.button("Predict"):
        result = model.predict(user_input)[0]
        result_text = "✅ Eligible" if result == 1 else "❌ Not Eligible"
        st.success(f"Prediction: **{result_text}**")

# Region filter
locations = sorted(farmer_df["Location"].unique())
selected_locations = st.multiselect("Select region(s)", locations, default=locations)

filtered_df = farmer_df[farmer_df["Location"].isin(selected_locations)]

# --------------------------------------------------------------------
# Charts
st.header("📈 Credit Score by Region", divider='gray')

region_chart = filtered_df.groupby("Location")["Credit Score"].mean().reset_index()
st.bar_chart(region_chart.set_index("Location"))

st.header("📊 Farmer Creditworthiness Table", divider='gray')
st.dataframe(filtered_df, use_container_width=True)

# --------------------------------------------------------------------
# Farmer performance metrics
st.header("📌 Key Metrics", divider='gray')

cols = st.columns(3)
for i, row in filtered_df.iterrows():
    col = cols[i % 3]
    with col:
        st.metric(
            label=f"Farmer {row['Farmer ID']}",
            value=f"{row['Credit Score']}",
            delta=f"{row['Loan Repayment %']}% Repayment",
            delta_color="normal" if row['Eligible'] else "inverse"
        )

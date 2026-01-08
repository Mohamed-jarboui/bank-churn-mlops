import streamlit as st
import requests
import json
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# API URL (Azure Production)
API_URL = "https://bank-churn.agreeablebay-fa140be5.swedencentral.azurecontainerapps.io"

st.title("🏦 Bank Customer Churn Predictor")
st.markdown("""
Cette application utilise un modèle de Machine Learning (Random Forest) déployé sur **Azure Container Apps** 
pour prédire la probabilité qu'un client quitte la banque.
""")

# Sidebar pour les informations du projet
with st.sidebar:
    st.header("Project Info")
    st.info(f"**API Status**: Connecting to {API_URL}")
    if st.button("🔍 Check API Health"):
        try:
            r = requests.get(f"{API_URL}/health")
            if r.status_code == 200:
                st.success("API is Healthy ✅")
            else:
                st.error(f"API Error: {r.status_code}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# Formulaire de saisie
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Profile")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Balance ($)", value=50000.0, step=1000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)

with col2:
    st.subheader("Activity & Geography")
    estimated_salary = st.number_input("Estimated Salary ($)", value=75000.0, step=1000.0)
    has_crcard = st.checkbox("Has Credit Card", value=True)
    is_active = st.checkbox("Is Active Member", value=True)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Encodage géographique (One-Hot)
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# Préparation de la requête
payload = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": 1 if has_crcard else 0,
    "IsActiveMember": 1 if is_active else 0,
    "EstimatedSalary": estimated_salary,
    "Geography_Germany": geo_germany,
    "Geography_Spain": geo_spain
}

# Bouton de prédiction
if st.button("🚀 Predict Churn Risk", use_container_width=True):
    with st.spinner("Analyzing customer data..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                prob = result["churn_probability"]
                risk = result["risk_level"]
                
                # Affichage des résultats
                st.divider()
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    color = "red" if risk == "High" else "orange" if risk == "Medium" else "green"
                    st.metric("Risk Level", risk, delta_color="inverse")
                    st.write(f"**Probability**: {prob*100:.2f}%")
                
                with res_col2:
                    # Jauge Plotly
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "wheat"},
                                {'range': [70, 100], 'color': "salmon"}
                            ],
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

# Section Monitoring
st.divider()
st.subheader("🛰️ Live Monitoring")
if st.button("📊 Trigger Data Drift Check"):
    with st.spinner("Comparing production vs training data..."):
        try:
            r = requests.post(f"{API_URL}/drift/check")
            if r.status_code == 200:
                res = r.json()
                if res["status"] == "success":
                    st.success(f"Drift Analysis Complete: {res['features_analyzed']} features analyzed.")
                else:
                    st.warning(res["message"])
            else:
                st.error(f"Drift check failed: {r.text}")
        except Exception as e:
            st.error(f"Drift check connection error: {e}")

st.caption("Bank Churn MLOps Workshop - Final Project")

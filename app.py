import streamlit as st
import numpy as np
import time as t
import matplotlib.pyplot as plt

st.set_page_config(page_title="FraudSense", page_icon="💳", layout="centered")

st.title("💳 FraudSense: FinTech Fraud Detection Demo")

st.divider()

st.subheader("📥 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    money = st.number_input("💰 Transaction Amount", min_value=0.0, format="%.2f", value=1200.00)
    time_input = st.number_input("⏰ Transaction Time (sec)", min_value=0.0, format="%.2f", value=180.00)
with col2:
    bank = st.selectbox("🏦 Bank Name", ["HDFC", "ICICI", "SBI", "Axis", "PNB", "Other"])
    location = st.text_input("📍 Location", "Mumbai")

st.divider()

st.subheader("🔍 Fraud Probability Prediction")

if st.button("Predict Fraud Risk"):
    with st.spinner("Analyzing transaction..."):
        t.sleep(1)

        # simple rule based fake probability
        fraud_prob = np.clip((money / 10000) + (time_input / 2000), 0, 1)

        st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")

        fig, ax = plt.subplots(figsize=(5, 0.5))
        ax.barh(["Fraud Risk"], [fraud_prob])
        ax.set_xlim(0, 1)
        st.pyplot(fig)

        if fraud_prob > 0.7:
            st.error("High Risk of Fraud Detected!")
        elif fraud_prob > 0.4:
            st.warning("Moderate Risk — manual check recommended.")
        else:
            st.success("Low Risk — looks legitimate.")

st.divider()

st.markdown("""
---
**Project:** FraudSense (Demo)  
**Developer:** Manya Gupta  
""")

import streamlit as st
import numpy as np
import time as t
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="FraudSense - Fraud Detection System", page_icon="💳", layout="centered")

# --- ATTENTION LAYER (same as training) ---
class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

# --- REBUILD MODEL + LOAD WEIGHTS ---
@st.cache_resource
def load_fraud_model():
    try:
        inputs1 = Input((1, 9))
        att_in = LSTM(50, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(inputs1)
        att_in_1 = LSTM(50, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(att_in)
        att_out = attention()(att_in_1)
        outputs1 = Dense(1, activation='sigmoid', trainable=True)(att_out)
        model = Model(inputs1, outputs1)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights("Save_Model_Attention.h5")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model weights: {e}")
        return None

model = load_fraud_model()

# --- HEADER ---
st.title("💳 FraudSense: FinTech Fraud Detection System")
st.markdown("""
This demo deploys **FraudSense**, a deep learning model (LSTM + Attention) that identifies potentially fraudulent transactions.  
Enter details below to simulate real-time detection.
""")

st.divider()

# --- INPUT SECTION ---
st.subheader("📥 Enter Transaction Details")

col1, col2 = st.columns(2)
with col1:
    money = st.number_input("💰 Transaction Amount", min_value=0.0, format="%.2f", value=1200.00)
    time_input = st.number_input("⏰ Transaction Time (in seconds)", min_value=0.0, format="%.2f", value=180.00)
with col2:
    bank = st.selectbox("🏦 Bank Name", ["HDFC", "ICICI", "SBI", "Axis", "PNB", "Other"])
    location = st.text_input("📍 Transaction Location (City/Area)", "Mumbai")

st.divider()

# --- SCENARIO-BASED FEATURE GENERATION ---
def get_demo_features(money, time_input, scenario="normal"):
    """Generate realistic standardized values for demo"""
    if scenario == "normal":  # safe
        return np.array([[[
            (money / 10000),
            (time_input / 1000),
            -0.22, 0.14, 0.09, -0.05, 0.08, 0.11, -0.12
        ]]])
    elif scenario == "medium":  # slightly unusual
        return np.array([[[
            (money / 10000) * 1.5,
            (time_input / 1000) * 0.8,
            0.31, -0.28, 0.22, 0.14, -0.25, 0.33, 0.17
        ]]])
    else:  # high-risk
        return np.array([[[
            (money / 10000) * 2.5,
            (time_input / 1000) * 1.2,
            0.55, -0.52, 0.48, 0.26, -0.33, 0.42, 0.35
        ]]])

# --- PREDICTION ---
st.subheader("🔍 Fraud Probability Prediction")

if st.button("Predict Fraud Risk"):
    with st.spinner("Analyzing transaction..."):
        t.sleep(1)

        if model is not None:
            # choose scenario based on amount
            if money < 1000:
                scenario = "normal"
            elif 1000 <= money < 5000:
                scenario = "medium"
            else:
                scenario = "high"

            input_features = get_demo_features(money, time_input, scenario)
            fraud_prob = float(model.predict(input_features)[0][0])

            st.caption(f"Scenario used: **{scenario.title()}** (auto-selected for demo realism)")
            st.metric(label="Fraud Probability", value=f"{fraud_prob*100:.2f}%")

            # --- FRAUD RISK BAR ---
            st.write("### Fraud Risk Level Visualization")
            fig, ax = plt.subplots(figsize=(5, 0.5))
            ax.barh(["Fraud Risk"], [fraud_prob],
                    color="red" if fraud_prob > 0.7 else "orange" if fraud_prob > 0.4 else "green")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

            # --- MESSAGE ---
            if fraud_prob > 0.7:
                st.error("High Risk of Fraud Detected!")
            elif fraud_prob > 0.4:
                st.warning("Moderate Risk: Needs manual verification.")
            else:
                st.success("Low Risk: Transaction appears legitimate.")

            # --- CONFIDENCE DISTRIBUTION (mock visual) ---
            st.write("### Model Confidence Distribution (Example)")
            random_probs = np.random.beta(2, 5, 100)  # skewed distribution
            fig2, ax2 = plt.subplots()
            ax2.hist(random_probs, bins=20, color='skyblue', edgecolor='black')
            ax2.axvline(fraud_prob, color='red', linestyle='--', label='Your Prediction')
            ax2.set_title("Distribution of Predicted Fraud Probabilities")
            ax2.set_xlabel("Probability")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.error("❌ Model not loaded. Please check weights file.")

st.divider()

# --- FOOTER ---
st.markdown("""
---
**Project:** FraudSense | **Developed by:** Manya Gupta & Tanmay Sharma  
**Model:** LSTM + Attention | **Deployment:** Streamlit
""")

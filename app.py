import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    model = joblib.load(BASE_DIR / "credit_card_fraud_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    return model, scaler


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #152944);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div.stButton > button {
    background-color: #14b8a6;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning Based Transaction Prediction")
st.divider()

st.subheader("🔍 Enter Transaction Details")

try:
    model, scaler = load_models()

    feature_count = getattr(
        scaler,
        "n_features_in_",
        getattr(model, "n_features_in_", None)
    )

    if feature_count is None:
        st.error("Could not determine the required input features.")
        st.stop()

    st.info(f"Model input features required: {feature_count}")

    with st.form("transaction_form"):
        values = []
        columns = st.columns(3)

        for i in range(feature_count):
            with columns[i % 3]:
                value = st.number_input(
                    f"Feature {i + 1}",
                    value=0.0,
                    format="%.6f",
                    key=f"feature_{i}"
                )
                values.append(value)

        submitted = st.form_submit_button(
            "🔎 Analyze Transaction"
        )

    if submitted:
        features = np.array(
            values, dtype=float
        ).reshape(1, -1)

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        st.subheader("Prediction Result")

        if int(prediction) == 1:
            st.error("⚠️ Prediction: Potential Fraudulent Transaction")
        else:
            st.success("✅ Prediction: Legitimate Transaction")

        st.caption(
            "This is a machine learning prediction, not a guarantee."
        )

except FileNotFoundError as e:
    st.error(f"Model file not found: {e.filename}")

except Exception as e:
    st.error(f"Error: {e}")

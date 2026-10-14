"""Streamlit interface for the Cuisine Classification project."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "cuisine_model.joblib"

st.set_page_config(page_title="Cuisine Classifier", page_icon="🍽️", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#101827 0%,#18283b 55%,#263c4b 100%); color:#f8fafc;}
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding: 1.6rem 1.8rem; border: 1px solid #3d596b; border-radius: 20px; background: rgba(255,255,255,.07); margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.6rem; margin-bottom: .3rem; color:#ffffff;}
.hero p {color:#cbd5e1; font-size:1.05rem;}
div[data-testid="stForm"] {padding: 1.2rem; border: 1px solid #3d596b; border-radius: 18px; background: rgba(255,255,255,.05);}
div[data-testid="stMetric"] {background: rgba(255,255,255,.07); border:1px solid #3d596b; padding:14px; border-radius:14px;}
</style>
<div class="hero"><h1>🍽️ Cuisine Classifier</h1><p>Machine Learning Task 3 · Cognifyz Technologies · Predict a restaurant's primary cuisine from restaurant details.</p></div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("Model file is missing. Run `python train_model.py` in the project folder first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
model = load_model()

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Restaurant details")
    with st.form("cuisine_form"):
        city = st.text_input("City", value="Bangalore", help="Enter a city represented in the dataset.")
        currency = st.selectbox("Currency", ["Indian Rupees(Rs.)", "Dollar($)", "Emirati Diram(AED)", "Brazilian Real(R$)", "Botswana Pula(P)", "Other"])
        country_code = st.number_input("Country code", min_value=0, max_value=999, value=1, step=1)
        cost = st.number_input("Average cost for two", min_value=0, max_value=100000, value=800, step=100)
        price_range = st.slider("Price range", min_value=1, max_value=4, value=2)
        rating = st.slider("Aggregate rating", min_value=0.0, max_value=5.0, value=3.5, step=0.1)
        votes = st.number_input("Number of votes", min_value=0, max_value=1000000, value=100, step=10)
        booking = st.selectbox("Table booking available?", ["No", "Yes"])
        delivery = st.selectbox("Online delivery available?", ["No", "Yes"])
        delivering = st.selectbox("Delivering now?", ["No", "Yes"])
        order_menu = st.selectbox("Switch to order menu?", ["No", "Yes"])
        submitted = st.form_submit_button("Predict cuisine", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")
    st.write("Fill in the restaurant details, then run the model to estimate the most likely primary cuisine.")
    st.info("This demo predicts the first cuisine label used in the training data. Predictions are estimates, not guarantees.")
    if submitted:
        row = pd.DataFrame([{
            "City": city.strip() or "Unknown", "Currency": "Other" if currency == "Other" else currency,
            "Has Table booking": booking, "Has Online delivery": delivery,
            "Is delivering now": delivering, "Switch to order menu": order_menu,
            "Country Code": country_code, "Average Cost for two": cost,
            "Price range": price_range, "Aggregate rating": rating, "Votes": votes
        }])
        try:
            pred = model.predict(row)[0]
            st.success(f"Predicted primary cuisine: **{pred}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(row)[0]
                classes = model.named_steps["classifier"].classes_
                top = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:5]
                st.markdown("#### Top model estimates")
                for name, probability in top:
                    st.write(f"**{name}** — {probability:.1%}")
                    st.progress(float(probability))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    else:
        st.markdown("### How it works")
        st.write("1. Enter restaurant attributes.\n2. The saved preprocessing pipeline handles missing values and categorical fields.\n3. Logistic Regression estimates the primary cuisine label.")

st.divider()
st.caption("Built with Python, pandas, scikit-learn and Streamlit. Model quality depends on the dataset and the features entered.")

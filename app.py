import streamlit as st
import numpy as np
import joblib

#loading model
model = joblib.load("LinearRegression_best_model.pkl")

# App title and logo
st.set_page_config(page_title="Tata Stock Close Price Predictor", page_icon="📈")
st.title("📊 Tata Stock Close Price Predictor")

st.image('TataLogo.jpg', caption='Tata Stocks Prediction', use_column_width=True)

# -------------------------
# Input Fields
# -------------------------
open_price = st.number_input("Open Price", value=0.0)
high_price = st.number_input("High Price", value=0.0)
low_price = st.number_input("Low Price", value=0.0)
vwap = st.number_input("The Volume Weighted Average Price (VWAP)", value=0.0)
trades = st.number_input("Trades", value=0.0)
turnover = st.number_input("Turnover", value=0.0)
ma_50 = st.number_input("50-Day Moving Average (MA_50)", value=0.0)

# -------------------------
# Prediction Logic
# -------------------------
if st.button("Predict Close Price"):
    input_data = np.array([[open_price, high_price, low_price, vwap, trades, turnover, ma_50]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"📈 Predicted Close Price: ₹{prediction:.2f}")

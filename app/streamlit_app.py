# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)


# Global weekday mapping (handles uppercase too)
weekday_map = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6
}

# UI Title
st.title("🏧 ATM Cash-Out Prediction System")
st.markdown("Predict how much cash to load in an ATM for tomorrow 💰")

# Mode selection
mode = st.radio("Choose Mode:", ["Use Latest Data (Auto)", "Enter Manually (Custom)"])

# --- Auto Mode ---
if mode == "Use Latest Data (Auto)":
    df = pd.read_csv("data/christ college atm.csv")
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], format="%d-%m-%Y")
    latest = df.sort_values("Transaction Date").iloc[-1]

    withdrawals = latest["No Of Withdrawals"]
    is_festival = 1 if latest["Festival Religion"] != "NH" else 0

    # Ensure weekday string is mapped properly
    weekday_str = str(latest["Weekday"]).lower().strip()
    day = weekday_map.get(weekday_str, 0)  # default to Monday if not matched

    st.info(f"Using last entry: {latest['Transaction Date'].date()}, "
            f"{withdrawals} withdrawals, Festival: {bool(is_festival)}, Day: {weekday_str.title()}")

# --- Manual Mode ---
else:
    withdrawals = st.number_input("Expected No. of Withdrawals", min_value=0, step=1)
    is_festival = st.selectbox("Is it a Festival Tomorrow?", ["No", "Yes"]) == "Yes"

    weekday = st.selectbox("Select Day of the Week", 
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    day = weekday_map[weekday.lower()]

# --- Prediction ---
if st.button("Predict Cash to Load"):
    input_data = [[withdrawals, int(is_festival), day]]
    pred = model.predict(input_data)[0]

    # Apply buffer: More cash for weekends or festivals
    buffer = 0.15 if is_festival or day in [5, 6] else 0.10
    to_load = pred * (1 + buffer)

    st.success(f"🔮 Predicted Withdrawals: ₹{pred:,.0f}")
    st.info(f"💰 Suggested Cash to Load: ₹{to_load:,.0f}")

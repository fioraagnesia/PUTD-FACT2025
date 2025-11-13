import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.graph_objects as go

# Load the saved Random Forest model and scaler
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to create radar chart
def radar_chart(data):
    categories = ["Timestamp (Month)", "Timestamp (Year)", "Gender", "Breed", "Weight (kg)", "Age of Chicken (Months)"]
    values = data.iloc[0].values

    # Adjust max values to include larger chicken ages
    min_values = [1, 2025, 0, 0, 0.0, 1]
    max_values = [12, 2035, 1, 2, 100.0, 36]  # Max age updated to 36 months

    normalized_values = [(v - min_val) / (max_val - min_val) for v, min_val, max_val in zip(values, min_values, max_values)]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    normalized_values = np.concatenate((normalized_values, [normalized_values[0]]))  # Close the circle
    angles += angles[:1]  # Close the circle

    fig, ax = plt.subplots(figsize=(6, 6), dpi=80, subplot_kw=dict(polar=True))
    ax.fill(angles, normalized_values, color='royalblue', alpha=0.3)
    ax.plot(angles, normalized_values, color='royalblue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color='darkblue')
    ax.set_title("Attributes Radar Chart", size=16, color="royalblue", weight='bold')
    return fig

# Function to calculate the best month to sell
def best_time_to_sell(rf_model, scaler, base_input):
    months = list(range(1, 13))
    predictions = []

    for month in months:
        base_input["Timestamp (Month)"] = month
        input_data = pd.DataFrame([base_input])
        input_data_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_data_scaled)[0]
        predictions.append(prediction)

    return months, predictions

# Function to plot best time to sell
def best_time_to_sell_plot(months, predictions):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
    ax.plot(months, predictions, marker='o', color='seagreen', label="Predicted Cost Price", linewidth=2)
    ax.set_xticks(months)
    ax.set_xticklabels(months, fontsize=12, fontweight='bold')
    ax.set_xlabel("Month", fontsize=14, fontweight='bold')
    ax.set_ylabel("Predicted Cost Price (IDR)", fontsize=14, fontweight='bold')
    ax.set_title("Best Time to Sell: Predicted Cost Price by Month", fontsize=16, fontweight='bold', color='seagreen')
    ax.legend()
    return fig

# Streamlit UI
def main():
    # Apply custom styling
    st.markdown("""
        <style>
            body {
                background-color: #f4f6f9;
                font-family: 'Helvetica', sans-serif;
            }
            .sidebar .sidebar-content {
                background-color: #f0f0f5;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            .stSlider>div>div>input {
                background-color: #e1f5fe;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and Sidebar Header
    st.title("Cost Price Prediction Dashboard", anchor="header")
    st.sidebar.header("Input Attributes")
    
    timestamp_year = 2025  # Fixed to current year
    timestamp_month = st.sidebar.slider("Timestamp (Month)", min_value=1, max_value=12, value=1)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    gender_value = 1 if gender == "Male" else 0
    breed = st.sidebar.selectbox("Breed", ["Broiler", "Layer", "Dual-purpose"])
    breed_map = {"Broiler": 2, "Layer": 1, "Dual-purpose": 0}
    breed_value = breed_map[breed]
    weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, max_value=100.0, value=1.0)

    # Adjusted slider for Age of Chicken
    age_of_chicken = st.sidebar.slider("Age of Chicken (Months)", min_value=1, max_value=36, value=6)

    # Input for cost to raise chicken
    cost_to_raise = st.sidebar.number_input("Cost to Raise Chicken (IDR)", min_value=0.0, value=50000.0)

    # Define input data
    base_input = {
        "Timestamp (Month)": timestamp_month,
        "Timestamp (Year)": timestamp_year,
        "Gender": gender_value,
        "Breed": breed_value,
        "Weight (kg)": weight,
        "Age of Chicken (Months)": age_of_chicken,
    }

    col1, col2 = st.columns([4,1])
    with col1:
        # Radar Chart Display
        input_data = pd.DataFrame([base_input])
        radar_fig = radar_chart(input_data)
        st.subheader("Attributes Radar Chart")
        st.pyplot(radar_fig)
    with col2:
        # Prediction
        input_data_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_data_scaled)
        predicted_cost = prediction[0]
        st.subheader("Predicted Cost Price")
        st.write(f"The predicted cost price is: **{predicted_cost:.2f} IDR**")

        # Calculate and display profit margin
        if cost_to_raise > 0:
            profit_margin = ((predicted_cost - cost_to_raise) / cost_to_raise) * 100
            st.subheader("Profit Margin")
            st.write(f"The profit margin is: **{profit_margin:.2f} %**")
        else:
            st.warning("Please enter a valid cost to raise the chicken.")

    # Best Time to Sell
    st.subheader("Best Time to Sell")
    months, predictions = best_time_to_sell(rf_model, scaler, base_input)
    best_time_fig = best_time_to_sell_plot(months, predictions)
    st.pyplot(best_time_fig)
    st.write(f"The best time to sell is **Month {months[np.argmax(predictions)]}** with the highest predicted cost price: **{max(predictions):.2f} IDR**")

if __name__ == '__main__':
    main()

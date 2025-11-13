import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle



# Function to create radar chart
def radar_chart(data):
    # Define the attributes
    categories = ["Physical Condition", "Gender", "Breed", "Feed"]
    values = data[["Physical Condition", "Gender", "Breed", "Feed"]].values.flatten()
    
    # Normalize values for radar chart scaling (0 to 1)
    values = (values - min(values)) / (max(values) - min(values))

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))  # To close the circle
    angles += angles[:1]  # To close the circle
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=80, subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)  # Line plot to outline
    ax.set_yticklabels([])  # Hide the radial ticks
    
    ax.set_xticks(angles[:-1])  # Set the labels at the angles
    ax.set_xticklabels(categories, fontsize=12)
    
    # Title
    ax.set_title("Attributes Radar Chart", size=14, color="blue", weight='bold')
    
    return fig

# Load the saved Random Forest model and scaler
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to create radar chart
def radar_chart(data):
    # Define the attributes
    categories = ["Physical Condition", "Gender", "Breed", "Feed"]
    values = data[["Physical Condition", "Gender", "Breed", "Feed"]].values.flatten()
    
    # Normalize values for radar chart scaling (0 to 1)
    values = (values - min(values)) / (max(values) - min(values))

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))  # To close the circle
    angles += angles[:1]  # To close the circle
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=80, subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)  # Line plot to outline
    ax.set_yticklabels([])  # Hide the radial ticks
    
    ax.set_xticks(angles[:-1])  # Set the labels at the angles
    ax.set_xticklabels(categories, fontsize=12)
    
    # Title
    ax.set_title("Attributes Radar Chart", size=14, color="blue", weight='bold')
    
    return fig

# Streamlit UI
def main():
    st.title("Profit Margin Prediction with Radar Chart")
    
    # Input fields for the user
    st.sidebar.header("Input Attributes")
    
    # Physical Condition input (mapping Grade A: 2, Grade B: 1, Grade C: 0)
    physical_condition = st.sidebar.selectbox("Physical Condition", ["Grade A", "Grade B", "Grade C"])
    condition_map = {"Grade A": 2, "Grade B": 1, "Grade C": 0}
    physical_condition_value = condition_map[physical_condition]
    
    # Gender input (Male: 1, Female: 0)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    gender_value = 1 if gender == "Male" else 0
    
    # Breed input (Broiler: 2, Layer: 1, Dual-purpose: 0)
    breed = st.sidebar.selectbox("Breed", ["Broiler", "Layer", "Dual-purpose"])
    breed_map = {"Broiler": 2, "Layer": 1, "Dual-purpose": 0}
    breed_value = breed_map[breed]
    
    # Feed input (Grains: 3, Forage: 2, Protein-rich feed: 1, Coconut residue: 0)
    feed = st.sidebar.selectbox("Feed", ["Grains", "Forage", "Protein-rich feed", "Coconut residue"])
    feed_map = {"Grains": 3, "Forage": 2, "Protein-rich feed": 1, "Coconut residue": 0}
    feed_value = feed_map[feed]
    
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[physical_condition_value, gender_value, breed_value, feed_value]],
                              columns=["Physical Condition", "Gender", "Breed", "Feed"])
    
    # Display radar chart in the center of the screen
    fig = radar_chart(input_data)
    st.pyplot(fig)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction using the Random Forest model
    prediction = rf_model.predict(input_data_scaled)
    
    # Display the prediction result
    st.subheader("Predicted Profit Margin (%)")
    st.write(f"The predicted profit margin is: {prediction[0]:.2f}%")

if __name__ == '__main__':
    main()

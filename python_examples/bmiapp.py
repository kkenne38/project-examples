import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(
    page_title="BMI Calculator",
    page_icon="⚖️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stRadio > label {
        font-weight: bold;
    }
    .result-text {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 BMI Calculator")
st.markdown("""
    Calculate your Body Mass Index (BMI) to check if you're at a healthy weight.
    BMI is a measure of body fat based on height and weight.
""")

# Create two columns for unit selection
col1, col2 = st.columns(2)

with col1:
    # Weight unit selection
    weight_unit = st.radio("Select Weight Unit", ["Kilograms", "Pounds"])

with col2:
    # Height unit selection
    height_unit = st.radio("Select Height Unit", ["Feet & Inches", "Meters"])

# Weight input
if weight_unit == "Kilograms":
    weight = st.number_input("Enter your weight (kg)", min_value=0.0, max_value=500.0, value=70.0, step=0.1)
else:
    weight = st.number_input("Enter your weight (lbs)", min_value=0.0, max_value=1000.0, value=154.0, step=0.1)
    weight = weight * 0.45359237  # Convert pounds to kg

# Height input
if height_unit == "Feet & Inches":
    col3, col4 = st.columns(2)
    with col3:
        feet = st.number_input("Feet", min_value=0, max_value=8, value=5, step=1)
    with col4:
        inches = st.number_input("Inches", min_value=0, max_value=11, value=7, step=1)
    height = (feet * 12 + inches) * 0.0254  # Convert to meters
else:
    height = st.number_input("Enter your height (m)", min_value=0.0, max_value=3.0, value=1.70, step=0.01)

# Calculate BMI when inputs are valid
if height <= 0:
    st.error("Height must be greater than 0")
elif weight <= 0:
    st.error("Weight must be greater than 0")
else:
    bmi = weight / (height ** 2)
    
    # Create columns for results
    col5, col6 = st.columns([2, 1])
    
    with col5:
        st.markdown("### Your BMI Result")
        st.markdown(f"<div class='result-text'>BMI: {bmi:.1f}</div>", unsafe_allow_html=True)
        
        # Determine BMI category
        if bmi < 18.5:
            category = "Underweight"
            color = "blue"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
            color = "green"
        elif 25 <= bmi < 30:
            category = "Overweight"
            color = "orange"
        else:
            category = "Obese"
            color = "red"
            
        st.markdown(f"<div style='color: {color};' class='result-text'>Category: {category}</div>", 
                   unsafe_allow_html=True)

    with col6:
        # Display BMI scale
        st.markdown("### BMI Scale")
        st.markdown("""
        - <18.5: Underweight
        - 18.5-24.9: Normal
        - 25-29.9: Overweight
        - ≥30: Obese
        """)

# Additional information
st.markdown("""
    ---
    ### Important Note
    BMI is a general indicator and may not be accurate for:
    - Athletes
    - Pregnant women
    - Elderly people
    - Children
    
    Always consult with a healthcare provider for proper health assessment.
""")

# Footer
st.markdown("""
    ---
    <div style='text-align: center; color: gray;'>
        Created with Streamlit • BMI Calculator v1.0
    </div>
""", unsafe_allow_html=True)
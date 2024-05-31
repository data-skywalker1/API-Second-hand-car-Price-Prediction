import pandas as pd
import streamlit as st
import datetime
import pickle

# Load the dataset and model
cars_df = pd.read_excel("cars24-car-price.xlsx")
with open("car_pred.pkl", "rb") as file:
    reg_model = pickle.load(file)

# the title and description
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# image related to second-hand cars
#st.image("car.jpg", use_column_width=True)

# logo at the top
#st.image("logo.png", width=100)  # Ensure the correct path to your logo file

st.title("Car Price Predictor")
st.markdown("### Get the best estimate for your used car!")

# Encoding dictionary
encode_dict = {
    "fuel_type": {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5},
    "seller_type": {'Dealer': 1, 'Individual': 2, 'Trustmark Dealer': 3},
    "transmission_type": {'Manual': 1, 'Automatic': 2}
}

def model_pred(fuel_type, transmission_type, engine, seats, reg_model):
    input_features = [[2018, 1, 4000, fuel_type, transmission_type, 19.7, engine, 86.3, seats]]
    return reg_model.predict(input_features)

# Layout
st.markdown("#### Please provide the following details:")

col1, col2 = st.columns(2)

with col1:
    fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
    engine = st.slider("Engine Power (cc)", 500, 5000, step=100)
    seats = st.selectbox("Number of Seats", [4, 5, 7, 9, 11])

with col2:
    transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Prediction button
if st.button("Predict Price"):
    fuel_type_encoded = encode_dict['fuel_type'][fuel_type]
    transmission_type_encoded = encode_dict["transmission_type"][transmission_type]
    price = model_pred(fuel_type_encoded, transmission_type_encoded, engine, seats, reg_model)
    
    st.success(f"🔮 Predicted Price of the car is: ₹{price[0]:,.2f} lakhs")

# Footer
st.markdown("---")
st.markdown("**Note:** This is a simulated prediction based on the given input features.")
st.markdown("Made with ❤️ by Anup")

# Add some styling
st.markdown("""
    <style>
        body {
            background-color: white; 
            color: white
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 16px;
        }
        .stMarkdown, .stImage {
            text-align: center;
        }
        .stTitle {
            color: #4CAF50;
        }
        .stSelectbox, .stSlider, .stButton {
            color: #000000;
        }
    </style>
    """, unsafe_allow_html=True)

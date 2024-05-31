import pandas as pd
import streamlit as st
import datetime
import pickle

# right now for the specific use case data is present as a static df, howver in actual sense it wil come from a db
cars_df=pd.read_excel("cars24-car-price.xlsx")

with open("car_pred.pkl","rb") as file:
    reg_model=pickle.load(file)

# the title and description
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")


st.title("Used Cars Price Prediction")
st.markdown("### Get the best estimate for your used car!")


encode_dict = {
    "fuel_type": {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5},
    "seller_type": {'Dealer': 1, 'Individual': 2, 'Trustmark Dealer': 3},
    "transmission_type": {'Manual': 1, 'Automatic': 2}
}

def model_pred(fuel_type,transmission_type,engine,seats,reg_model):
    
    
        input_features=[[2018,1,4000,fuel_type,transmission_type,19.7,engine,86.3,seats]]

        return(reg_model.predict(input_features))
    
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

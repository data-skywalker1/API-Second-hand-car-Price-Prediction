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
    
        # here I am taking some features in coded manner but we can take them from user
        input_features=[[2019,1,4000,fuel_type,transmission_type,19.7,engine,86.3,seats]]

        return(reg_model.predict(input_features))
    
col1, col2,col3= st.columns(3)

fuel_type=col1.selectbox("Select the fuel type",["Diesel","Petrol","CNG","LPG","Electric"])
engine=col1.slider("Set the engine power",500,5000,step=100)

transmission_type=col3.selectbox("Select the transmission type",["Manual","Automatic"])

seats=col1.selectbox("Enter the number of seats",[4,5,7,9,11])
random=col3.slider("Set the random power",500,5000,step=100)
# random2=col3.selectbox("Enter random2 seats",[4,5,7,9,11])

if (st.button("Predict Price")):
    fuel_type=encode_dict['fuel_type'][fuel_type]
    transmission_type=encode_dict["transmission_type"][transmission_type]
    price=model_pred(fuel_type,transmission_type,engine,seats,reg_model)
    st.text("Predicted Price of the car is:"+ str(price))



    # Footer
st.markdown("---")
st.markdown("**Note:** This is a simulated prediction based on the given input features.")
st.markdown("Made with ❤️ by Anup")

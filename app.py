import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
from xgboost import XGBRegressor
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'xgb_model.joblib')

#page settings
st.set_page_config(page_title="Site EUI Prediction App",
                   page_icon="⚡", layout="wide")

#page header
st.title(f'Site EUI Predictor App')

#Creating option list for dropdown menu
facility_type_options = ['Grocery_store_or_food_market','Commercial_Other','Industrial','Office_Uncategorized','Health_Care_Inpatient',
                            'Multifamily_Uncategorized','Education_Other_classroom','Laboratory','Warehouse_Nonrefrigerated','Lodging_Hotel']
building_class_option = ['Residential','Commercial']
state_options = ['State_1','State_2','State_3','State_4','State_5','State_7','State_8','State_9','State_10','State_11']

features = ['STATE_FACTOR', 'building_class', 'facility_type','year_built', 'energy_star_rating', 'building_area', 'floor_area', 'avg_temp']

st.markdown("<h1 style='text-align:center;'>Site EUI Predictor App⚡</h1>",unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):
        st.subheader("Enter the input for following features:")
        
        STATE_FACTOR = st.selectbox("Select State:",options=state_options)
        building_class = st.selectbox("Select Building Class:",options=building_class_option)
        facility_type = st.selectbox("Select Facility Type:",options=facility_type_options)
        year_built = st.slider("Pickup Year:",0,2015,value=0,format="%d")
        energy_star_rating = st.slider("Pickup Rating:",0,100,value=0,format="%d")
        building_area = st.slider("Pickup Building Area:",0,600000000,value=0,format="%d")
        floor_area = st.slider("Pickup Floor Area:",0,6000000,value=0,format="%d")
        avg_temp = st.slider("Pickup Rating:",0,70,value=0,format="%d")

        submit = st.form_submit_button("Predict")

    if submit:
        STATE_FACTOR = ordinal_encoder(STATE_FACTOR, state_options)
        building_class = ordinal_encoder(building_class, building_class_option)
        facility_type = ordinal_encoder(facility_type, facility_type_options)
        
        data = np.array([STATE_FACTOR,building_class,facility_type,year_built,energy_star_rating,building_area,floor_area,avg_temp]).reshape(1,-1)
        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()
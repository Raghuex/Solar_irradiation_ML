import xgboost as xgb
import streamlit as st 
import numpy as np
import pandas as pd

st.title("Solar Irradiation Prediction")


  

st.info('This is a prediction model based on XGBoost Regressor')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('cleaned_data.csv')
  df

  st.write('**X**')
  X_raw = df.drop(columns=[df.columns[0], 'Radiation'], axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.Radiation
  y_raw

#Input Features
with st.sidebar:
  st.header("Input Features")
  Pressure = st.number_input('Pressure (inHg)', min_value=30.4, max_value=30.6, value=30.5)
  Humidity = st.number_input('Humidity ', min_value=0, max_value=100)
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))

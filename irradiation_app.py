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

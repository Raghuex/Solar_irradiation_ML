import os
import pickle
import xgboost as xgb
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler



# loading the saved model
model_xgb = pickle.load(open("solar_xgb.pkl", "rb"))


st.title("Solar Irradiation Prediction")

st.info('This is a prediction model based on XGBoost Regressor')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('cleaned_data.csv')
  df

  st.write('**X**')
  X = df.drop(columns=[df.columns[0], 'Radiation'], axis=1)
  X

  st.write('**y**')
  y = df.Radiation
  y

#Input Features
with st.sidebar:
  st.header("Input Features")
  pressure = st.number_input('Pressure', min_value=30.4, max_value=30.6, value=30.5)
  temperature = st.number_input('Temperature', min_value=30, max_value=80, value=55)
  humidity = st.number_input('Humidity', min_value=0, max_value=100, value=50)
  wind_direction = st.number_input('WindDirection(Degrees)', min_value=0, max_value=360, value=180)
  speed = st.number_input('Speed', min_value=0, max_value=42, value=21)
  month = st.slider('Month', min_value=9, max_value=12, value=9)
  day = st.number_input('Day', min_value=1, max_value=31, value=15)
  hour = st.number_input('Hour', min_value=0, max_value=23, value=6)
  minute = st.number_input('Minute', min_value=0, max_value=59, value=0)
  second = st.number_input('Second', min_value=0, max_value=59, value=0)
  rise_hour = st.number_input('Rise Hour', value=6, disabled=True)
  rise_minute = st.number_input('Rise Minute', min_value=0, max_value=59, value=0)
  set_hour = st.selectbox('Set Hour', options=[17, 18], index=0)
  set_minute = st.number_input('Set Minute', min_value=0, max_value=59, value=0)


# Create a dictionary with all input features
data = {
    'pressure': pressure,
    'temperature': temperature,
    'humidity': humidity,
    'wind_direction': wind_direction,
    'speed': speed,
    'month': month,
    'day': day,
    'hour': hour,
    'minute': minute,
    'second': second,
    'rise_hour': rise_hour,
    'rise_minute': rise_minute,
    'set_hour': set_hour,
    'set_minute': set_minute
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame([data])

with st.expander('Input features'):
  st.write('**Input for Solar Radiation**')
  input_df








# prediction _________________________


prediction = model_xgb.predict(input_df)
st.write('**Predicted Solar Radiation**:', prediction[0])



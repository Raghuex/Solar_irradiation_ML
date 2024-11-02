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


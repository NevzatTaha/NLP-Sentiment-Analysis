import pandas as pd
import numpy as np
import re 
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,classification_report
import joblib
import streamlit as st
import plotly.express as px



df=pd.read_csv("Cleaned.csv",encoding='latin1')
df=df.set_index(df['textID'])
df=df.drop(columns='Unnamed: 0')


st.set_page_config( page_title="Sentiment Analysis and Dashbord", page_icon="🤗",layout="wide")
st.title("Data Science - :red[Sentiment Analysis] :sunglasses:")


st.subheader('Dataframe')
st.write(df[['text', 'sentiment',
       'Time of Tweet', 'Age of User', 'Country', 'Population -2020',
       'Land Area (KmÂ²)', 'Density (P/KmÂ²)']])

sidebar=st.sidebar.selectbox(label="Content",options=("Country Information","Age Information","Model Predicton"))

locations='https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes'

iso_alpha=pd.read_html(locations,match='Name[5]')

iso_alpha[0]

df["iso_alpha"]=

if sidebar=="Country Information":
    px.scatter_geo(data_frame=df,locations='Country')
    






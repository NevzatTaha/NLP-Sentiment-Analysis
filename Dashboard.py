import pandas as pd
import numpy as np
import re 
import joblib
import streamlit as st
import plotly.express as px
import pycountry
import seaborn as sns
import matplotlib.pyplot as plt


# Data Cleaning
df=pd.read_csv("Cleaned.csv",encoding='latin1')
df=df.set_index(df['textID'])
df=df.drop(columns='Unnamed: 0')


# Data Preparation 
iso_alpha_dict = {country.name: country.alpha_3 for country in pycountry.countries}

country_to_iso_dict = {country: iso_alpha_dict.get(country, None) for country in df['Country']}

df['ISO_alpha'] = df['Country'].map(country_to_iso_dict)


number_of_tweets = df.groupby('Country').size().reset_index(name='Number of Tweets')

df = pd.merge(df, number_of_tweets, on='Country')


nlp=joblib.load('finalmodel.pkl')

             


# Deployement

st.set_page_config( page_title="Sentiment Analysis and Dashbord", page_icon="🤗",layout="wide")
st.title("Data Science - :red[Sentiment Analysis] :sunglasses:") # Head of the Website and its title



sidebar=st.sidebar.selectbox(label="Content",options=("Main Page","Country Information","Age Information","Model Prediction"))


if sidebar=="Model Predicton":
       st.subheader("The model purposed to understand sentiment of the tweets. It gives whether your tweets are positive, neutral or negative.")
       text=st.text_area("Please write a tweet that you want to learn its sentiment 🤗")
       result=nlp.predict([text])
       if st.button("Lets Predict🤗") and len(text)>1:
              if result=='positive':
                     st.write("This is a positive tweet :sunglasses:")
              elif result=='neutral':
                     st.write("This is a  neutral tweet 😐")
              elif result=='negative':
                     st.write("This is a negative tweet 😕")
              else:
                     st.write('There is a mistake') 
       else:
              st.subheader("Please write a tweet.")
elif sidebar == "Main Page":
    st.subheader('Dataframe')
    st.write(df[['text', 'sentiment',
       'Time of Tweet', 'Age of User', 'Country', 'Population -2020',
       'Land Area (KmÂ²)', 'Density (P/KmÂ²)']])
elif sidebar == "Country Information":
       st.subheader('This is world map that shows number of tweets and the size of the countries.')
       earth_map = px.scatter_geo(data_frame=df, locations='ISO_alpha',color="Land Area (KmÂ²)",hover_name='Country', size="Number of Tweets", projection="natural earth")
       earth_map.update_layout(width=1000)
       st.plotly_chart(earth_map)
elif sidebar=="Age Information":
       fig = px.histogram(df, x="Age of User", color="sentiment", barmode="group", histfunc="count", 
                       category_orders={"sentiment": ["positive", "neutral", "negative"]}, 
                       color_discrete_sequence=px.colors.qualitative.Set2)
       fig.update_layout(width=1000)
       # st.pyplot(barplot2.figure)
       st.plotly_chart(fig)
       st.write("In every age of user , there are similar patters because of good distributed training data.")      
        

              
              







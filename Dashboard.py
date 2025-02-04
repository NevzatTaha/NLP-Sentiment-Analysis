import pandas as pd
import numpy as np
import re 
import joblib
import streamlit as st
import plotly.express as px
import pycountry
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
import time
from wordcloud import WordCloud

st.set_page_config( page_title="Sentiment Analysis and WordCloud", page_icon="🤗",layout="wide")


# Cache the NLTK downloads and model loading
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    return SnowballStemmer(language='english')

stemmer = load_nltk()

# Cache the tokenization function
@st.cache_data
def tokenizaton_stemming(text):
    tokenization = nltk.word_tokenize(text)
    return [stemmer.stem(tokens) for tokens in tokenization]

# Cache word frequency calculation
@st.cache_data(show_spinner=False)
def frequency_of_words(df, labels, number_of_words=20, _tokenize=None):
    frequencies = {}
    for label in labels:
        cv = CountVectorizer(stop_words='english', tokenizer=_tokenize)
        matrix = cv.fit_transform(df[df['sentiment'] == label]['text'].values.astype('U'))
        freqs = zip(cv.get_feature_names_out(), matrix.sum(axis=0).tolist()[0])
        frequencies[label] = sorted(freqs, key=lambda x: -x[1])[:number_of_words]
    return frequencies


    ...

# Cache data loading and preparation
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("Cleaned.csv", encoding='latin1')
    df = df.set_index('textID')
    df = df.drop(columns='Unnamed: 0')
    
    iso_alpha_dict = {country.name: country.alpha_3 for country in pycountry.countries}
    country_to_iso_dict = {country: iso_alpha_dict.get(country, None) for country in df['Country']}
    df['ISO_alpha'] = df['Country'].map(country_to_iso_dict)
    
    number_of_tweets = df.groupby('Country').size().reset_index(name='Number of Tweets')
    return pd.merge(df, number_of_tweets, on='Country')

# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load('finalmodel.pkl')

# Load data and model
df = load_and_prepare_data()
nlp = load_model()

# Calculate word frequencies once
word_frequencies = frequency_of_words(df, ['positive', 'negative', 'neutral'], _tokenize=tokenizaton_stemming)
positive_frequencies = word_frequencies.get('positive', [])
neutral_frequencies = word_frequencies.get('neutral', [])
negative_frequencies = word_frequencies.get('negative', [])

# Cache WordCloud generation
@st.cache_data
def generate_wordcloud(frequencies, **kwargs):
    return WordCloud(width=600, height=400, background_color='white', stopwords="english", **kwargs).generate_from_frequencies(dict(frequencies))

# Deployement


st.title("NLP - Sentiment Analysis ☁️☁️") # Head of the Website and its title


with st.sidebar:
       st.title('X Tweets Sentiment Analysis ☁️')


sidebar=st.sidebar.selectbox(label='',options=("Model Predicton","Data Frame","WordCloud"))


if sidebar=="Model Predicton":
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
              st.success('Congrats!! you have just learned your tweets sentiment and it is {results} tweet'.format(results=result))
       
       with st.expander('About', expanded=True):
        st.write('''
            - :orange[**Linkedn**]: https://www.linkedin.com/in/nevzatayhan/.
            - :orange[**GitHub**]: https://github.com/NevzatTaha
            - For any cooperations or suggestions please send an Email: Nevtahaayhan@gmail.com
            ''')
elif sidebar == "Data Frame":
    st.subheader('Dataframe')
    st.write(df[['text', 'sentiment',
       'Time of Tweet', 'Age of User', 'Country', 'Population -2020',
       'Land Area (KmÂ²)', 'Density (P/KmÂ²)']])
    with st.expander('Details',expanded=True ):
           st.write('''
                    - :orange[**Resource of the Project**]: https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data)
                    ''')
# elif sidebar == "Country Information":
#        st.subheader('This is world map that shows number of tweets and the size of the countries.')
#        earth_map = px.scatter_geo(data_frame=df, locations='ISO_alpha',color="Land Area (KmÂ²)",hover_name='Country', size="Number of Tweets", projection="natural earth")
#        earth_map.update_layout(width=1000)
#        st.plotly_chart(earth_map)

elif sidebar=="WordCloud":# Generate WordCloud
              st.subheader("These are words cloud that explains which words are most used for the specific labels.")
              
              # Generate WordClouds
              wordcloud_positive = generate_wordcloud(positive_frequencies)
              wordcloud_neutral = generate_wordcloud(neutral_frequencies)
              wordcloud_negative = generate_wordcloud(negative_frequencies, colormap='cool')
              
              # Display WordClouds
              st.image(wordcloud_positive.to_array(), caption='Word Cloud for Positive Reviews')
              st.image(wordcloud_neutral.to_array(), caption='Word Cloud for Neutral Reviews')
              st.image(wordcloud_negative.to_array(), caption='Word Cloud for Negative Reviews')

        

              
              







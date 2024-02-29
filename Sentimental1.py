import pandas as pd
import numpy as np
import re 


# Reading Data
df=pd.read_csv('C:/Users/nevta/Desktop/NLP Sentimental/NLP-Sentiment-Analysis/train.csv',encoding='latin1')

df_test=pd.read_csv('C:/Users/nevta/Desktop/NLP Sentimental/NLP-Sentiment-Analysis/test.csv',encoding='latin1')


df_=df.copy()

df_test_=df_test.copy()

df.info()

# Data Cleaning
df.isna().sum();

df=df.dropna(axis=0) # Text was null, so it not usefull data. Drop it.

# df_test.isna().sum();


# df=df.dropna(how='all',axis=0);

# df_test=df_test.dropna(how='all',axis=0);
# ### Some Rows are completely null.


def text_clean(text):
    text = re.sub(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',repl= '',string= text)  # For the mail
    text = re.sub(pattern=r'\b(?:https?://)\S+\b', repl= '',string= text)  # For the URL
    text = re.sub(pattern=r'\bhttps?://\S+\b', repl= '',string= text)  # For the URL
    text = re.sub(pattern=r'[-/_\']', repl= '',string= text)  
    text = re.sub(pattern='[0-9]', repl= '',string= text)   # For removing numbers
    return(text) 
 
df["text"]=df["text"].apply(text_clean)



df["text"]=df["text"].str.strip()### For the white space

###  For the all same country, population, land are and density have to be same

country_mistake=[]
for country in df['Country'].unique():
    if df[df['Country']==country]['Population -2020'].nunique()!=1:
         country_mistake.append(country)
         country_mistake.append('Population -2020')
    elif df[df['Country']==country]['Land Area (Km²)'].nunique()!=1:
         country_mistake.append(country)
         country_mistake.append('Land Area (Km²)')
    elif df[df['Country']==country]['Density (P/Km²)'].nunique()!=1:
         country_mistake.append(country)
         country_mistake.append('Density (P/Km²)')

### country_mistake it is an empthy, so it is good. We looked whether there is a diverstiy for the certain data.

        
    
# Machine Learning Sentiment Analysis

X_train=df['text']
y_train=df['sentiment']

X_test=df_test['text']
y_test=df_test['sentiment']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer


 cv=CountVectorizer(stop_words='english'
                    )

tfidf_transformer = TfidfTransformer()








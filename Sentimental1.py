import pandas as pd
import numpy as np
import re 


# Reading Data
df=pd.read_csv('test.csv',encoding='latin1')

df_=df.copy()
df

df.info()

# Data Cleaning
df.isna().sum();


df=df.dropna(how='all',axis=0);
### Some Rows are completely null.


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
for country in df.columns:
    if df[df['Country']=='Venezuela']['Population -2020'].nunique()!=1:
         country_mistake.append(country)

### country_mistake it is an empthy, so it is good

        
    
# Machine Learning Sentiment Analysis

X_train=df['text']
y=df['']










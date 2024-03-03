import pandas as pd
import numpy as np
import re 
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


# Reading Data
df=pd.read_csv('train.csv',encoding='latin1')
df_test=pd.read_csv('test.csv',encoding='latin1')
df_=df.copy()
df_test_=df_test.copy()
df.info()

# Data Cleaning
df.isna().sum();

df=df.dropna(axis=0) # Text was null, so it not usefull data. Drop it.

df_test.isna().sum();


df=df.dropna(how='all',axis=0);

df_test=df_test.dropna(how='all',axis=0);
 # # # # # #Some Rows are completely null


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
# # Test and Train data
X_train=df['text']
y_train=df['sentiment']

X_test=df_test['text']
y_test=df_test['sentiment']



# Model Selection 

# # # LinearSVC()

pipe=Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('svc',LinearSVC())]) 

pipe.fit(X_train,y_train)

pred_Linear=pipe.predict(X_test)



print(classification_report(y_test,pred_Linear))
print(accuracy_score(y_test,pred_Linear))



# # # Naive Bayes()



pipe_NB=Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('nb',MultinomialNB())]) 

pipe_NB.fit(X_train,y_train)

y_predict=pipe_NB.predict(X_test)


print(classification_report(y_test,y_predict))
print(accuracy_score(y_test,y_predict))

# # # Logistic Regression
pipe_lr=Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('lr', LogisticRegression(max_iter=1000))]) 

pipe_lr.fit(X_train,y_train)

y_predict_lr=pipe_lr.predict(X_test)


print(classification_report(y_test,y_predict_lr))
print(accuracy_score(y_test,y_predict_lr))
# # # Hence LogisticRegression is the best algorithm for the projet


# Load model

joblib.dump(pipe_lr,'finalmodel.pkl')


test=joblib.load('final_model.pkl')

test

prediction = test.predict(['It is awesome'])

print(prediction)

# For the analysis, I will extract data.

df_=df_.dropna()

df_.to_csv('Cleaned.csv')
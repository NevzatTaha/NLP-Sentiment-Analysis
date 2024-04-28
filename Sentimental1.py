import pandas as pd
import numpy as np
import re 
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import nltk
from nltk.stem import SnowballStemmer


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

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
    text = re.sub(pattern=r'https\S+|www\S+|https\S+', repl= ' ',string= text)  # For the URL
    text = re.sub(pattern=r'(\\d|\\W)+', repl= ' ',string= text)#Remove digits and non-word characters (\W).
    text = re.sub(pattern=r'\@\w+|\#', repl= ' ',string= text)  #Remove Twitter handles (@username) and hashtags (#).
    text = re.sub(pattern=r'[^\w\s\`]', repl= ' ',string= text)  #Remove any characters that are not alphanumeric, whitespace, or a backtick (`).
    text = re.sub(pattern='[0-9]', repl= ' ',string= text)  # For the number
    text = re.sub(r'\b\w\b',repl= ' ',string= text) #For the single character
    text = re.sub(pattern=r'[^\w\s]', repl=' ', string=text)  # Remove backtick from the pattern
    return(text) 

df["text"]=df["text"].apply(text_clean)





df["text"]=df["text"].str.strip()### For the white space

#Stemming
nltk.download('punkt')  # Ensure you have punkt tokenizer downloaded
stemmer = SnowballStemmer(language='english')
def tokenizaton_stemming(text):
     tokenization=nltk.word_tokenize(text)
     stemming=  [stemmer.stem(tokens)  for tokens in tokenization]
     return stemming

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



#  Controlling Most 20 words

def frequency_of_words(labels, number_of_words=20, tokenize=None):
    for label in labels:
        cv = CountVectorizer(stop_words='english', tokenizer=tokenize)
        matrix = cv.fit_transform(df[df['sentiment'] == label]['text'])
        freqs = zip(cv.get_feature_names_out(), matrix.sum(axis=0).tolist()[0])
        # sort from largest to smallest
        print(f"Top {number_of_words} words used for {label} reviews.")
        print(sorted(freqs, key=lambda x: -x[1])[:number_of_words])

frequency_of_words(['positive', 'negative', 'neutral'], tokenize=tokenizaton_stemming)

# # If the 20 words are made up nonsense, you have to control your data cleaning process


    
# Machine Learning Sentiment Analysis
# # Test and Train data
X_train=df['text']
y_train=df['sentiment']

X_test=df_test['text']
y_test=df_test['sentiment']



# Model Selection 

# # # LinearSVC()
models={LinearSVC():'svc',LogisticRegression(max_iter=1000):'lr',MultinomialNB():'nb',KNeighborsClassifier():'knn',GradientBoostingClassifier():'gb'}




def model_selection(expected_model):
 # Please import expected model before the run the function
     for model,shortcuts in expected_model.items():
          pipe=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',tokenizer=tokenizaton_stemming)),(shortcuts,model)])
          pipe.fit(X_train,y_train)
          y_predict=pipe.predict(X_test)
          print(f'{shortcuts} Results',classification_report(y_test,y_predict))
          print(f'{shortcuts} Results',accuracy_score(y_test,y_predict))

model_selection(models)

# LogisticRegression has the most accuracy.Now lets set the hyperparameters
pipe_lr=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',tokenizer=tokenizaton_stemming)),('lr', LogisticRegression(max_iter=10000,class_weight='balanced',C=1))]) 

pipe_lr.fit(X_train,y_train)

y_predict_lr=pipe_lr.predict(X_test)


print(classification_report(y_test,y_predict_lr))
print(accuracy_score(y_test,y_predict_lr))








# Load model
joblib.dump(pipe_lr,'finalmodel.pkl')




# Test
model = joblib.load('finalmodel.pkl')
# Now, you can use the loaded model for prediction
prediction = model.predict(['It is awesome'])

print(prediction)













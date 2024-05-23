# NLP Sentiment-Analysis
## Purpose
Dataset has different variables. In this project, I have improved a NLP project with machine learning. The challenge is to create a model to predict tweets sentiments.

## Steps

### Data Cleaning
There are different formats and words that I should not want to add into my model. For instance, I want to remove stop words or numbers, such as ?,!,3,5. Also, I want to extract their origin because 
words can be varied via prefix or suffix. Therefore I have used stemming.
![image](https://github.com/NevzatTaha/NLP-Sentiment-Analysis/assets/108625825/91169b1f-db49-495e-9259-b6d948ae6d66)

![image](https://github.com/NevzatTaha/NLP-Sentiment-Analysis/assets/108625825/66460b9d-0978-48a8-ad99-e10e3f441a44)

### Data Visualisation
Word Cloud is the most efficient way to visualize nlp projects. Hence, I extracted most used 20 words. During the deployement, I have made it for the each label.

![image](https://github.com/NevzatTaha/NLP-Sentiment-Analysis/assets/108625825/8b75aa3d-c4e6-4103-93bf-8ede953b4116)

![image](https://github.com/NevzatTaha/NLP-Sentiment-Analysis/assets/108625825/8a8430d1-db3d-4485-bb15-9f80bf1ad403)


### Model Training
There are many model that can be used. Therefore, I have written a function which applies models for my data. Then I took the one which shows highest most accurate one.
![image](https://github.com/NevzatTaha/NLP-Sentiment-Analysis/assets/108625825/99d1667a-5ad6-44e0-a6b4-e754cb8132a7)

### Model Dump
I dumped the model via joblib.

### Web Application and Deployement
I designed simple and understandable pages for users. There are three different pages that shows three most important parts of the project. 
![image](https://github.com/NevzatTaha/NLP-Sentiment-Analysis/assets/108625825/af0f757f-681d-4a55-bed5-b5d08c49817c)

![image](https://github.com/NevzatTaha/NLP-Sentiment-Analysis/assets/108625825/ea60750a-73c7-419c-b1da-22eace70c4c8)

![image](https://github.com/NevzatTaha/NLP-Sentiment-Analysis/assets/108625825/d959c9da-256e-4ecd-9b0a-c5da2193a4bf)

### Web Application 
https://dashboardpy-fedqetb5ndpdftthv6uxu2.streamlit.app






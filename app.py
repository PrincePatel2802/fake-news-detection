from flask import Flask
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')

# Initialize
ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# Load your dataset (replace with your actual path)
@flask.cache_data
def load_data():
    df = pd.read_csv("train.csv")  # Make sure the CSV has 'text' and 'label' columns
    df = df.dropna()
    df['text'] = df['text'].apply(preprocess_text)
    return df

# Train model
@flask.cache_resource
def train_model(data):
    X = data['text']
    y = data['label']
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(X).toarray()
    model = LogisticRegression()
    model.fit(X, y)
    return model, tfidf

# Streamlit UI
def main():
    flask.title("📰 Fake News Detection using AI")

    menu = ["Home", "Detect"]
    choice = flask.sidebar.selectbox("Menu", menu)

    data = load_data()
    model, tfidf = train_model(data)

    if choice == "Home":
        flask.subheader("Project Overview")
        flask.write("This app detects whether a news article is **Fake** or **Real** using a Machine Learning model.")
        flask.write("Model: Logistic Regression")
        flask.write("Dataset used: [train.csv] with columns 'text' and 'label' (0 = Real, 1 = Fake)")

    elif choice == "Detect":
        flask.subheader("Enter News Text")
        input_news = flask.text_area("Type or paste the news content here")

        if flask.button("Check"):
            if input_news.strip() == "":
                flask.warning("Please enter some text!")
            else:
                processed = preprocess_text(input_news)
                vectorized = tfidf.transform([processed]).toarray()
                prediction = model.predict(vectorized)[0]
                result = "Fake News ❌" if prediction == 1 else "Real News ✅"
                flask.success(f"Prediction: **{result}**")

if __name__ == '__main__':
   main()

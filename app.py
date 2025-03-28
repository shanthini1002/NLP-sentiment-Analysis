import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
def load_data():
    df = pd.read_csv("/content/amazon_alexa.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
    return df

df = None

# Sidebar Navigation
st.sidebar.title("Sentiment Analysis App")
option = st.sidebar.radio("Select Section:", ["Load Dataset", "EDA", "Data Preprocessing", "Visualizations", "Model Training & Evaluation"])

# Load Dataset Section
if option == "Load Dataset":
    st.title("Load Dataset")
    df = load_data()
    st.write("Dataset Loaded Successfully!")
    st.write(df.head())

# EDA Section
elif option == "EDA":
    st.title("Exploratory Data Analysis")
    if df is None:
        df = load_data()
    st.write(df.head())
    st.write(df.describe())
    st.write("### Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())

# Data Preprocessing
elif option == "Data Preprocessing":
    st.title("Data Preprocessing")
    if df is None:
        df = load_data()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(text):
        if pd.isnull(text) or not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    
    df['cleaned_reviews'] = df['verified_reviews'].fillna('').apply(preprocess_text)
    st.write("Text Preprocessing Completed")
    st.write(df[['verified_reviews', 'cleaned_reviews']].head())

# Visualization
elif option == "Visualizations":
    st.title("Data Visualization")
    if df is None:
        df = load_data()
    text = ' '.join(df['verified_reviews'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
    st.write("### Sentiment Distribution")
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df['sentiment'], palette='viridis')
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")
    st.pyplot(plt)

# Model Training & Evaluation
elif option == "Model Training & Evaluation":
    st.title("Model Training & Evaluation")
    if df is None:
        df = load_data()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_reviews'])
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Naïve Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear')
    }
    
    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        st.write(f"**{name} Accuracy:** {acc:.4f}")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))
    
    st.write("### Model Accuracy Comparison")
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='viridis')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    st.pyplot(plt)

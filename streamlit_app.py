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

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", [
    "Load Dataset", "EDA", "Data Preprocessing", "Visualizations", "Model Training & Evaluation", "Predict Sentiment"
])

# File uploader
st.title("Upload File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Multi-class sentiment labeling
    def label_sentiment(rating):
        if rating <= 2:
            return 0  # Negative
        elif rating == 3:
            return 1  # Neutral
        else:
            return 2  # Positive

    df['sentiment'] = df['rating'].apply(label_sentiment)

    # Preprocessing
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

    # Feature extraction
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_reviews'])
    y = df['sentiment']

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define models
    models = {
        "Naïve Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
        "SVM": SVC(kernel='linear', probability=True, class_weight='balanced')
    }

    # Train models and store results
    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']),
            "conf_matrix": confusion_matrix(y_test, y_pred)
        }

    # Section-wise logic
    if option == "Load Dataset":
        st.header("Dataset Loaded")
        st.write(df.head())
        st.success("Dataset Loaded Successfully!")
    
    elif option == "EDA":
        st.header("Exploratory Data Analysis")
        st.subheader("Basic Info")
        st.write(df.info())
        st.write("Shape:", df.shape)
        st.write("Missing values:")
        st.write(df.isnull().sum())
        st.write("Sentiment Distribution:")
        st.write(df['sentiment'].value_counts())

    elif option == "Data Preprocessing":
        st.header("Data Preprocessing")
        st.write(df[['verified_reviews', 'cleaned_reviews']].head())

    elif option == "Visualizations":
        st.header("Visualizations")

        st.subheader("Rating Distribution")
        sns.countplot(x='rating', data=df, palette='Set1')
        st.pyplot(plt)

        st.subheader("Sentiment Distribution")
        plt.figure(figsize=(6, 4))
        sns.countplot(x='sentiment', data=df, palette='Set2')
        plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'])
        st.pyplot(plt)

        st.subheader("Boxplot of Ratings")
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df['rating'], color='orange')
        st.pyplot(plt)

        st.subheader("Word Cloud")
        text = ' '.join(df['verified_reviews'].fillna('').astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    elif option == "Model Training & Evaluation":
        st.header("Model Training & Evaluation")

        accuracies = {name: result["accuracy"] for name, result in model_results.items()}
        for name, result in model_results.items():
            st.subheader(name)
            st.write(f"Accuracy: {result['accuracy']:.2f}")
            st.text("Classification Report:\n" + result["report"])

        st.subheader("Model Accuracy Comparison")
        plt.figure(figsize=(8, 5))
        sorted_acc = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))
        sns.barplot(x=list(sorted_acc.keys()), y=list(sorted_acc.values()), palette='viridis')
        plt.ylabel("Accuracy")
        st.pyplot(plt)

    elif option == "Predict Sentiment":
        st.header("Predict Sentiment")
        user_input = st.text_area("Enter review text:")

        if st.button("Analyze"):
            if user_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                model = model_results["SVM"]["model"]  # Default to SVM
                prediction = model.predict(vectorizer.transform([user_input]))[0]
                label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                st.write(f"**Predicted Sentiment:** {label_map[prediction]}")

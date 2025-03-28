
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
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", [
    "Load Dataset", "EDA", "Data Preprocessing", "Visualizations", "Model Training & Evaluation","Predict Sentiment"])

st.title("Upload File")
uploaded_file = st.file_uploader("Choose a file", type=["csv"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip() 

# Load dataset
    df['date'] = pd.to_datetime(df['date'])
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
# data preprocessing
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
# model building and evaluation
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
    

if option == "Load Dataset":
    st.title("Load Dataset")
    st.write(df)
    st.write("Dataset Loaded Successfully!")
    

# EDA Section
elif option == "EDA":
    st.title("Exploratory Data Analysis")
    st.write("### First 5 rows")
    st.write(df.head())
    st.write("### Data Info")
    st.write(df.info())
    st.write("### Number of rows and columns")
    st.write(df.shape)
    st.write("### Data types")
    st.write(df.dtypes)
    st.write("### Checking for missing values")
    st.write(df.isnull().sum())
# Data Preprocessing
elif option == "Data Preprocessing":
    st.title("Data Preprocessing")
    st.write("Text Preprocessing Completed")
    st.write(df[['verified_reviews', 'cleaned_reviews']].head())

# Visualization
elif option == "Visualizations":
    st.title("Data Visualization")
    st.write("### Distribution of Ratings")
    sns.countplot(x='rating', data=df,color="green")
    st.pyplot(plt)

    st.write("### Sentiment Distribution")
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df['sentiment'], palette='viridis')
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    st.pyplot(plt)

    st.write("### Boxplot to identify outliers")
    plt.figure(figsize=(10, 6))
    sns.boxplot(df,color="orange")  
    st.pyplot(plt)
    
    text = ' '.join(df['verified_reviews'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.write("### Wordcloud")
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
    

# Model Training & Evaluation
elif option == "Model Training & Evaluation":
    st.title("Model Training & Evaluation")
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
        st.write(f"**{name}**")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))
    
    st.write("### Model Accuracy Comparison")
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='viridis')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    st.pyplot(plt)
# Sentiment Prediction
elif option == "Predict Sentiment":
    st.title("Predict Sentiment")
    user_input = st.text_area("Enter text to analyze sentiment:")
    if st.button("Analyze"):
        model = models["SVM"]  # Default model
        prediction = model.predict(vectorizer.transform([user_input]))
        
        # Check if the prediction is a probability instead of a class label
        if isinstance(prediction[0], float):  # In case prediction is a probability
            if prediction[0] > 0.5:  # Adjust the threshold as needed
                sentiment = "Positive"
            else:
                sentiment = "Negative"
        else:  # If the prediction is a class label (0 or 1)
            if prediction[0] == 1:
                sentiment = "Positive"
            else:
                sentiment = "Negative"
        
        st.write(f"Predicted Sentiment: {sentiment}")


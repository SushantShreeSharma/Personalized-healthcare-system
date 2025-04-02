import pandas as pd
import numpy as np
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load_data():
    url = "https://drive.google.com/uc?id=1kV6WZ4-1uwwbIW1eESOsNGtZa4Nlw1SE"
    df = pd.read_csv(url, delimiter=",", on_bad_lines="skip", encoding="utf-8")
    df.columns = df.columns.str.strip()  # Remove extra spaces from column names
    df.drop(columns=['index'], inplace=True, errors='ignore')
    df.dropna(how="any", inplace=True)  # Remove rows with missing values
    return df

# Preprocess data
def preprocess_data(df):
    df.drop_duplicates(inplace=True)
    return df

# Train recommendation system
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Description'].fillna(""))
    return vectorizer, tfidf_matrix

# Recommend medicines
def recommend_medicine(condition, df, vectorizer, tfidf_matrix):
    condition_vector = vectorizer.transform([condition])
    similarity_scores = cosine_similarity(condition_vector, tfidf_matrix).flatten()
    recommendations = df.loc[np.argsort(similarity_scores)[-5:][::-1]]
    return recommendations[['Drug_Name', 'Reason', 'Description']]

# Streamlit UI
def main():
    st.title("Personalized Healthcare System")
    df = load_data()
    df = preprocess_data(df)
    vectorizer, tfidf_matrix = train_model(df)
    
    user_input = st.text_input("Enter a condition (e.g., Acne, Diabetes)")
    if st.button("Find Medicine") and user_input:
        results = recommend_medicine(user_input, df, vectorizer, tfidf_matrix)
        st.write(results)

if __name__ == "__main__":
    main()


import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Sentence Similarity Checker")

sentence1 = st.text_input("Enter the first sentence:")
sentence2 = st.text_input("Enter the second sentence:")

if st.button("Compare"):
    if sentence1 and sentence2:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([sentence1, sentence2])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
        st.success(f"Cosine Similarity Score: {similarity[0][0]:.4f}")
    else:
        st.warning("Please enter both sentences.")

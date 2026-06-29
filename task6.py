import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("YoutubeCommentsDataSet.csv")

df = df.head(3000)

df.columns = df.columns.str.strip()

df["Comment"] = df["Comment"].fillna("")

stop_words = ENGLISH_STOP_WORDS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_text"] = df["Comment"].apply(clean_text)

tfidf = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 1)
)

tfidf_matrix = tfidf.fit_transform(df["clean_text"])

def recommend(comment_text, top_n=5):

    matches = df[df["Comment"] == comment_text]

    if len(matches) == 0:
        return ["Comment not found"]

    idx = matches.index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]

    recommendations = []

    for i in similar_indices:
        recommendations.append(
            df.iloc[i]["Comment"]
        )

    return recommendations

st.set_page_config(
    page_title="YouTube Comment Recommendation System",
    layout="wide"
)

st.title("YouTube Comment Recommendation System")

st.write(
    "Select a comment and get similar comments based on TF-IDF and Cosine Similarity."
)

selected_comment = st.selectbox(
    "Select a Comment",
    df["Comment"]
)

if st.button("Get Recommendations"):

    recommendations = recommend(
        selected_comment,
        top_n=5
    )

    st.subheader("Recommended Similar Comments")

    for i, rec in enumerate(recommendations, start=1):
        st.write(f"{i}. {rec}")

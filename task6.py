import streamlit as st
import pandas as pd
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("YoutubeCommentsDataSet.csv")

df.columns = df.columns.str.strip()

stop_words = set(stopwords.words("english"))

df["Comment"] = df["Comment"].fillna("")

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
    max_features=5000,
    ngram_range=(1,2)
)

tfidf_matrix = tfidf.fit_transform(df["clean_text"])

similarity_matrix = cosine_similarity(tfidf_matrix)

def recommend(comment_text, top_n=5):

    idx = df[
        df["Comment"] == comment_text
    ].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x:x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n+1]

    results = []

    for i, score in similarity_scores:

        results.append(
            df.iloc[i]["Comment"]
        )

    return results

st.title("YouTube Comment Recommendation System")

selected_comment = st.selectbox(
    "Select a Comment",
    df["Comment"]
)

if st.button("Get Recommendations"):

    recommendations = recommend(
        selected_comment
    )

    st.subheader(
        "Recommended Similar Comments"
    )

    for i, rec in enumerate(
        recommendations,
        start=1
    ):
        st.write(f"{i}. {rec}")
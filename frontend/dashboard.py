import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"

st.title("AI Sentiment Analysis Dashboard")

st.write("Enter a review and the AI will detect the sentiment.")

review = st.text_area("Enter your review")

if st.button("Analyze Sentiment"):

    response = requests.post(API_URL, json={"text": review})
    result = response.json()

    sentiment = result["sentiment"]
    confidence = result["confidence"]
    emotion = result["emotion"]

    if sentiment == "POSITIVE":
        st.success(f"Sentiment: {sentiment}")

    elif sentiment == "NEGATIVE":
        st.error(f"Sentiment: {sentiment}")

    else:
        st.warning(f"Sentiment: {sentiment}")

    st.write("Confidence:", confidence)
    st.write("Emotion:", emotion)

st.subheader("Sentiment Distribution")

st.subheader("Batch Review Analysis")

reviews = st.text_area("Enter multiple reviews (one per line)")

if st.button("Analyze Reviews"):

    review_list = reviews.split("\n")

    results = []

    for r in review_list:

        if r.strip() != "":

            response = requests.post(API_URL, json={"text": r})
            result = response.json()

            results.append({
                "Review": r,
                "Sentiment": result["sentiment"],
                "Confidence": result["confidence"]
            })

    df = pd.DataFrame(results)

    st.dataframe(df)

    # Dynamic chart connected to AI results
    st.subheader("Sentiment Distribution")

    sentiment_counts = df["Sentiment"].value_counts()

    fig = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title="Sentiment Distribution from AI Analysis"
    )

    st.plotly_chart(fig)

    st.subheader("Word Cloud from Reviews")

    text = " ".join(review_list)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    fig_wc, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig_wc)

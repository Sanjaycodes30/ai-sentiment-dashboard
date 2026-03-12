
from model.sentiment_model import analyze_sentiment

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.title("AI Sentiment Analysis Dashboard")

st.write("Enter a review and the AI will detect the sentiment.")

review = st.text_area("Enter your review")

if st.button("Analyze Sentiment"):

    result = analyze_sentiment(review)

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

            result = analyze_sentiment(r)

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

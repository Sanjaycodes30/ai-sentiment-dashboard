from transformers import pipeline
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_aspects(text):

    doc = nlp(text)

    aspects = []

    for token in doc:
        if token.pos_ == "NOUN":
            aspects.append(token.text.lower())

    return list(set(aspects))


def analyze_aspects(text):

    aspects = extract_aspects(text)

    aspect_results = []

    for aspect in aspects:

        sentiment_result = sentiment_pipeline(text)[0]

        aspect_results.append({
            "aspect": aspect,
            "sentiment": sentiment_result["label"]
        })

    return aspect_results


sentiment_pipeline = pipeline("sentiment-analysis")

emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)


def analyze_sentiment(text):

    sentiment_result = sentiment_pipeline(text)[0]

    emotion_result = emotion_pipeline(text)[0][0]

    aspects = analyze_aspects(text)

    return {
        "sentiment": sentiment_result["label"],
        "confidence": float(sentiment_result["score"]),
        "emotion": emotion_result["label"],
        "aspects": aspects
    }

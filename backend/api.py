from fastapi import FastAPI
from pydantic import BaseModel
from model.sentiment_model import analyze_sentiment

app = FastAPI()


class Review(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.post("/predict")
def predict(review: Review):
    result = analyze_sentiment(review.text)
    return result

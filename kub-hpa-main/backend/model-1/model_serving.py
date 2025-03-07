from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.get("/results")
def get_results():
    df = pd.read_csv("stock_sentimentScore.csv")
    return df.to_dict(orient="records")

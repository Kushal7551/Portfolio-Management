import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import sys

import requests
import pandas as pd
from datetime import datetime, timedelta

# input_data = json.loads(sys.stdin.read())

api_key = "dbe0bf42faf44f41917591471d452cdf"  


stocks = {
    "BLUEDART": "Blue Dart Express",
    "NHPC": "NHPC Limited",
    "ADANIPOWER": "Adani Power"
}


end_date = datetime.now()
start_date = end_date - timedelta(days=7)
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")


def fetch_headlines(stock, query):
    url = (
        f"https://newsapi.org/v2/everything?q={query}&from={start_date_str}&to={end_date_str}"
        f"&sortBy=relevancy&language=en&apiKey={api_key}"
    )
    response = requests.get(url)
    articles = response.json().get("articles", [])
    headlines = [{"headline": article["title"], "stock": stock} for article in articles]
    return headlines


all_headlines = []
for stock, query in stocks.items():
    headlines = fetch_headlines(stock, query)
    all_headlines.extend(headlines)


df = pd.DataFrame(all_headlines, columns=["headline", "stock"])
# df.to_csv("stock_headlines.csv", index=False)

print("Dataset created with headlines for the past week.")


headlines_df = df
headlines_list = list(headlines_df['headline'])


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


batch_size = 10  


positive_scores = []
negative_scores = []
neutral_scores = []


for i in range(0, len(headlines_list), batch_size):
    batch = headlines_list[i:i + batch_size]
    
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    positive_scores.extend(predictions[:, 0].tolist())
    negative_scores.extend(predictions[:, 1].tolist())
    neutral_scores.extend(predictions[:, 2].tolist())


result_df = pd.DataFrame({
    'Headline': headlines_list,
    'Positive': positive_scores,
    'Negative': negative_scores,
    'Neutral': neutral_scores
})


result_df.to_csv("stock_sentimentScore.csv", index=False)
# print(json.dumps(5))

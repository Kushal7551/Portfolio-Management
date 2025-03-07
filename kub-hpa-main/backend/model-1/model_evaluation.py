import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("finbert_model")
model = AutoModelForSequenceClassification.from_pretrained("finbert_model")

# Load the dataset
df = pd.read_csv("stock_headlines.csv")
headlines = df['headline'].tolist()

batch_size = 10
positive_scores = []
negative_scores = []
neutral_scores = []

for i in range(0, len(headlines), batch_size):
    batch = headlines[i:i + batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_scores.extend(predictions[:, 0].tolist())
    negative_scores.extend(predictions[:, 1].tolist())
    neutral_scores.extend(predictions[:, 2].tolist())

df['Positive'] = positive_scores
df['Negative'] = negative_scores
df['Neutral'] = neutral_scores
df.to_csv("stock_sentimentScore.csv", index=False)
print("Model evaluation complete.")

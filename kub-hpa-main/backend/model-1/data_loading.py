import pandas as pd
import requests
from datetime import datetime, timedelta

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
    return [{"headline": article["title"], "stock": stock} for article in articles]

all_headlines = []
for stock, query in stocks.items():
    all_headlines.extend(fetch_headlines(stock, query))

df = pd.DataFrame(all_headlines, columns=["headline", "stock"])
df.to_csv("stock_headlines.csv", index=False)
print("Data loading complete.")

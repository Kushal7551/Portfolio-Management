from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Save the model for later use
model.save_pretrained("finbert_model")
tokenizer.save_pretrained("finbert_model")
print("Model training complete.")

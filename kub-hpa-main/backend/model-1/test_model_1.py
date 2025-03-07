import pytest
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from model_serving import app

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Create a TestClient for the FastAPI app

# Sample data for testing
sample_headlines = [
    "The company's stock price surged after the positive earnings report.",
    "A recent scandal has negatively affected the company's reputation.",
    "The market remained neutral with no significant changes reported."
]


@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame for testing."""
    return pd.DataFrame({"headline": sample_headlines, "stock": ["TEST1", "TEST2", "TEST3"]})


def test_data_loading(sample_dataframe):
    """Test if the data loading step creates a valid DataFrame."""
    assert not sample_dataframe.empty, "DataFrame is empty."
    assert "headline" in sample_dataframe.columns, "'headline' column is missing."
    assert "stock" in sample_dataframe.columns, "'stock' column is missing."


def test_model_inference():
    """Test if the model processes input correctly and returns expected output shapes."""
    inputs = tokenizer(sample_headlines, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    assert outputs.logits.shape[0] == len(sample_headlines), "Output batch size does not match input batch size."
    assert outputs.logits.shape[1] == 3, "Output does not contain three sentiment classes."


def test_softmax_output():
    """Test if the model's output scores sum to approximately 1 (softmax property)."""
    inputs = tokenizer(sample_headlines, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    for scores in predictions:
        assert abs(scores.sum().item() - 1.0) < 1e-6, "Softmax scores do not sum to 1."



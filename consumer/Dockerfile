FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download sentiment model weights and tokenizer into the image to avoid runtime downloads.
# Uses the default model defined in the consumer code.
RUN python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
PY

COPY . .

CMD ["python", "es_consumer.py"]

FROM python:3.9-slim

WORKDIR /app

COPY search_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY search_api/search_api.py .

# Copy frontend files
COPY frontend/index.html /frontend/
COPY frontend/topics.html /frontend/

CMD ["python", "search_api.py"]

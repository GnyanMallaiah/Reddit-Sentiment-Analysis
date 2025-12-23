# Real-Time Sentiment Analysis of Reddit Comments

This repository provides a containerized, real-time pipeline to ingest Reddit posts, run BERT/RoBERTa-based sentiment analysis, and index results into Elasticsearch for search and visualization.

**Architecture**: Reddit API (PRAW) → Kafka Producer → Kafka Broker → Kafka Consumer (BERT-based) → Elasticsearch → Search API + Dashboard

**Key Features**:
- Real-time sentiment analysis using Twitter-RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment-latest)
- Kafka-based event streaming for scalability
- Elasticsearch with BM25 search and aggregations
- Interactive web dashboard with real-time statistics
- Docker Compose orchestration for easy deployment

## Quickstart (Docker Compose)

1. Copy `.env.example` to `.env` and fill in your Reddit API credentials and any overrides.

2. Build and start the stack:

```bash
docker compose up --build
```

3. Services:
- **Search API + Dashboard**: http://localhost:8000
- **Kibana**: http://localhost:5601
- **Elasticsearch**: http://localhost:9200
- **Kafka broker**: localhost:9092

4. Access the web dashboard at http://localhost:8000 or use Kibana at http://localhost:5601 to explore and visualize the data.

## Project Structure

```
├── producer/          # Reddit API client → Kafka producer
├── consumer/          # Kafka consumer → Sentiment analysis → Elasticsearch
├── search_api/        # FastAPI search endpoint and web dashboard
├── frontend/          # Dashboard HTML/CSS/JS
├── docker-compose.yml # Complete stack orchestration
└── .env.example       # Configuration template
```

## Configuration

- **Sentiment Model**: Default is `cardiffnlp/twitter-roberta-base-sentiment-latest` (trained on 124M tweets)
- **Reddit Credentials**: Set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` in `.env`
- **Search Customization**: Modify search_api/search_api.py for custom query logic

## Features

- **Real-time Processing**: Sub-second latency from Reddit post to indexed document
- **Advanced Search**: Multi-word queries with phrase matching and relevance boosting
- **Sentiment Aggregations**: Distribution charts, time-series analysis, topic clustering
- **Filters**: Search by subreddit, sentiment label, topics, date range
- **Scalable Architecture**: Kafka enables horizontal scaling of consumers



          

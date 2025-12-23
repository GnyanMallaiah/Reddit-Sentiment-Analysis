import os
import json
import time
from dotenv import load_dotenv
from kafka import KafkaConsumer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from elasticsearch import Elasticsearch, helpers
from langdetect import detect, LangDetectException

load_dotenv()

KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'reddit_posts')
ES_HOST = os.getenv('ELASTICSEARCH_HOST', 'http://elasticsearch:9200')
ES_INDEX = os.getenv('ES_INDEX', 'reddit_sentiment')
MODEL_NAME = os.getenv('SENTIMENT_MODEL', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
ZERO_SHOT_MODEL = os.getenv('ZERO_SHOT_MODEL', 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')

# Reddit's official topic categories with keywords
TOPIC_KEYWORDS = {
    'sports': ['sport', 'game', 'team', 'player', 'football', 'basketball', 'baseball', 'soccer', 'hockey', 'tennis', 'golf', 'racing', 'fitness', 'workout', 'gym', 'nfl', 'nba', 'mlb'],
    'gaming': ['game', 'gaming', 'play', 'gamer', 'xbox', 'playstation', 'nintendo', 'steam', 'pc gaming', 'console', 'esports', 'fps', 'rpg', 'mmo'],
    'news': ['news', 'breaking', 'update', 'report', 'announce', 'headline', 'current events', 'world news'],
    'entertainment': ['movie', 'film', 'tv', 'show', 'series', 'netflix', 'actor', 'actress', 'celebrity', 'music', 'song', 'album', 'concert', 'band'],
    'technology': ['tech', 'technology', 'computer', 'software', 'hardware', 'app', 'phone', 'iphone', 'android', 'ai', 'programming', 'code', 'developer', 'startup'],
    'business': ['business', 'company', 'startup', 'entrepreneur', 'finance', 'stock', 'market', 'invest', 'economy', 'corporate'],
    'education': ['education', 'school', 'college', 'university', 'student', 'teacher', 'learn', 'study', 'course', 'degree'],
    'science': ['science', 'research', 'study', 'scientific', 'experiment', 'biology', 'chemistry', 'physics', 'space', 'nasa'],
    'health': ['health', 'fitness', 'medical', 'doctor', 'hospital', 'medicine', 'wellness', 'nutrition', 'diet', 'mental health'],
    'food': ['food', 'recipe', 'cooking', 'eat', 'restaurant', 'drink', 'coffee', 'beer', 'wine', 'chef'],
    'travel': ['travel', 'trip', 'vacation', 'tour', 'visit', 'destination', 'hotel', 'flight', 'airport'],
    'nature': ['animal', 'nature', 'wildlife', 'pet', 'dog', 'cat', 'bird', 'environment', 'climate'],
    'art': ['art', 'artist', 'creative', 'design', 'draw', 'paint', 'photo', 'photography', 'music']
}


def connect_es(max_retries=5, delay=5):
    for i in range(max_retries):
        try:
            es = Elasticsearch(ES_HOST)
            if es.ping():
                print('Connected to Elasticsearch')
                return es
        except Exception as e:
            print(f'Elasticsearch connection failed (attempt {i+1}): {e}')
        time.sleep(delay)
    raise RuntimeError('Failed to connect to Elasticsearch')


def init_model():
    print(f'Loading sentiment model: {MODEL_NAME}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, return_all_scores=False)
    # warm-up
    try:
        sentiment('This is a test')
    except Exception:
        pass
    print('Sentiment model loaded successfully', flush=True)
    return sentiment


def classify_topics_keyword(text, max_topics=2):
    """Fast keyword-based topic classification.
    
    Args:
        text: Text to classify
        max_topics: Maximum number of topics to return
    
    Returns:
        List of topic labels based on keyword matches
    """
    if not text:
        return ['general']
    
    text_lower = text.lower()
    topic_scores = {}
    
    # Score each topic based on keyword matches
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            topic_scores[topic] = score
    
    # Get top topics
    if not topic_scores:
        return ['general']
    
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, _ in sorted_topics[:max_topics]]


def preprocess_text(text):
    if not text:
        return ''
    new_text = []
    for t in text.split(' '):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return ' '.join(new_text)


def main():
    es = connect_es()
    sentiment = init_model()

    print('Connecting to Kafka...', flush=True)
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    print('Connected to Kafka successfully', flush=True)

    bulk_ops = []
    BATCH_SIZE = int(os.getenv('ES_BULK_SIZE', '100'))
    SENTIMENT_BATCH_SIZE = 10  # Process sentiment in batches of 10
    message_buffer = []  # Buffer messages for batch sentiment analysis

    print('Starting Kafka consumer loop...', flush=True)
    try:
        for msg in consumer:
            data = msg.value

            # Combine title and text
            title = (data.get('title') or '').strip()
            text = (data.get('text') or '').strip()
            
            # If text is empty, use title as text for better searchability
            if not text and title:
                text = title
            
            combined = f"{title} {text}".strip()

            # Language filter
            try:
                lang = detect(combined) if combined else 'en'
            except LangDetectException:
                lang = 'unknown'

            if lang != 'en':
                continue

            processed = preprocess_text(combined)
            if not processed:
                continue

            raw_flair = data.get('link_flair_text') or None
            
            # Fast keyword-based topic classification
            topics = classify_topics_keyword(combined)

            # Add to message buffer
            message_buffer.append({
                'data': data,
                'title': title,
                'text': text,
                'combined': combined,
                'processed': processed,
                'raw_flair': raw_flair,
                'topics': topics
            })

            # Process batch when buffer is full
            if len(message_buffer) >= SENTIMENT_BATCH_SIZE:
                # Batch sentiment analysis
                processed_texts = [m['processed'][:512] for m in message_buffer]
                try:
                    batch_sentiments = sentiment(processed_texts)
                except Exception as e:
                    print(f'Batch sentiment error: {e}', flush=True)
                    batch_sentiments = [{'label': 'neutral', 'score': 0.5}] * len(message_buffer)
                
                # Create documents for all messages in batch
                for idx, msg_data in enumerate(message_buffer):
                    label = batch_sentiments[idx].get('label', 'neutral').lower()
                    score = float(batch_sentiments[idx].get('score', 0.5))
                    
                    doc = {
                        '_index': ES_INDEX,
                        '_id': msg_data['data'].get('id'),
                        '_source': {
                            'id': msg_data['data'].get('id'),
                            'created_at': int(msg_data['data'].get('created_at')) if msg_data['data'].get('created_at') else None,
                            'subreddit': msg_data['data'].get('subreddit'),
                            'title': msg_data['title'],
                            'text': msg_data['text'],
                            'raw_flair': msg_data['raw_flair'],
                            'topics': msg_data['topics'],
                            'ingested_at': int(time.time()),
                            'sentiment_label': label,
                            'sentiment_score': score
                        }
                    }
                    bulk_ops.append(doc)
                
                # Clear message buffer
                message_buffer = []

            # Bulk index to Elasticsearch when bulk_ops is full
            if len(bulk_ops) >= BATCH_SIZE:
                try:
                    helpers.bulk(es, bulk_ops)
                    print(f'Indexed {len(bulk_ops)} documents', flush=True)
                except Exception as e:
                    print(f'Bulk index error: {e}', flush=True)
                bulk_ops = []

    except KeyboardInterrupt:
        print('Shutting down consumer...')
    finally:
        # Process any remaining buffered messages
        if message_buffer:
            processed_texts = [m['processed'][:512] for m in message_buffer]
            try:
                batch_sentiments = sentiment(processed_texts)
            except:
                batch_sentiments = [{'label': 'neutral', 'score': 0.5}] * len(message_buffer)
            
            for idx, msg_data in enumerate(message_buffer):
                label = batch_sentiments[idx].get('label', 'neutral').lower()
                score = float(batch_sentiments[idx].get('score', 0.5))
                
                doc = {
                    '_index': ES_INDEX,
                    '_id': msg_data['data'].get('id'),
                    '_source': {
                        'id': msg_data['data'].get('id'),
                        'created_at': int(msg_data['data'].get('created_at')) if msg_data['data'].get('created_at') else None,
                        'subreddit': msg_data['data'].get('subreddit'),
                        'title': msg_data['title'],
                        'text': msg_data['text'],
                        'raw_flair': msg_data['raw_flair'],
                        'topics': msg_data['topics'],
                        'ingested_at': int(time.time()),
                        'sentiment_label': label,
                        'sentiment_score': score
                    }
                }
                bulk_ops.append(doc)
        
        # Index any remaining documents
        if bulk_ops:
            try:
                helpers.bulk(es, bulk_ops)
                print(f'Indexed remaining {len(bulk_ops)} documents', flush=True)
            except Exception as e:
                print(f'Final bulk index error: {e}', flush=True)
        consumer.close()


if __name__ == '__main__':
    main()

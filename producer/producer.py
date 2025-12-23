import os
import json
import time
import praw
from kafka import KafkaProducer
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

# Load environment variables
load_dotenv()

# Reddit API credentials
reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_user_agent = os.getenv('REDDIT_USER_AGENT')

# Kafka configuration
kafka_topic = os.getenv('KAFKA_TOPIC')
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS')

# Subreddits to track. If `SUBREDDITS` is empty or set to 'all', stream from all subreddits.
subreddits_env = os.getenv('SUBREDDITS', '').strip()
if not subreddits_env or subreddits_env.lower() == 'all':
    subreddits = None
else:
    subreddits = [s.strip() for s in subreddits_env.split(',') if s.strip()]

def connect_to_reddit():
    """Connect to Reddit API"""
    return praw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent
    )

def connect_to_kafka():
    """Connect to Kafka and return producer"""
    return KafkaProducer(
        bootstrap_servers=kafka_bootstrap_servers,
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )

def stream_reddit_data(reddit, producer):
    """Stream Reddit data and send to Kafka, skipping blank and non-English posts"""
    # Use Reddit "all" if subreddits is None (meaning user requested all subreddits)
    if subreddits is None:
        subreddit = reddit.subreddit('all')
    else:
        subreddit = reddit.subreddit("+".join(subreddits))
    
    # Process submissions
    for submission in subreddit.stream.submissions(skip_existing=True):
        title = (submission.title or "").strip()
        selftext = (submission.selftext or "").strip()
        
        # Skip if both title and selftext are blank
        if not title and not selftext:
            continue
        
        # Combine title and selftext to detect language
        text_to_check = f"{title} {selftext}".strip()
        
        try:
            language = detect(text_to_check)
        except LangDetectException:
            continue  # Skip if language detection fails
        
        if language != 'en':
            continue  # Skip non-English posts
        
        data = {
            'id': submission.id,
            'created_at': submission.created_utc,
            'subreddit': submission.subreddit.display_name,
            'link_flair_text': submission.link_flair_text,
            'title': title,
            'text': selftext,
            'type': 'submission'
        }
        producer.send(kafka_topic, value=data)
        print(f"Sent submission data: {title} | {selftext}")
        producer.flush()



def main():
    """Main function to stream Reddit data to Kafka"""
    # Connect to Reddit API
    print("Connecting to Reddit API...")
    reddit = connect_to_reddit()
    
    # Wait for Kafka to be ready
    max_retries = 30
    retries = 0
    while retries < max_retries:
        try:
            print(f"Connecting to Kafka (attempt {retries+1}/{max_retries})...")
            producer = connect_to_kafka()
            break
        except Exception as e:
            print(f"Failed to connect to Kafka: {e}")
            retries += 1
            time.sleep(10)
    
    if retries == max_retries:
        print("Failed to connect to Kafka after multiple attempts. Exiting.")
        return
    
    if subreddits is None:
        print("Connected to Kafka. Streaming data from: all subreddits")
    else:
        print(f"Connected to Kafka. Streaming data from subreddits: {', '.join(subreddits)}")
    
    try:
        stream_reddit_data(reddit, producer)
    except KeyboardInterrupt:
        print("Stopping Reddit stream...")
    except Exception as e:
        print(f"Error in streaming: {e}")
    finally:
        if producer:
            producer.close()
            print("Kafka producer closed.")

if __name__ == "__main__":
    main()
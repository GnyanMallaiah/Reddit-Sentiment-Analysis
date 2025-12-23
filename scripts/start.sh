
#!/usr/bin/env bash
set -euo pipefail

# Start Zookeeper and Kafka first, wait until Kafka is reachable on localhost:9092,
# then start the remaining services (Elasticsearch, Kibana, producer, consumer).

echo "Starting Zookeeper and Kafka..."
docker compose up -d zookeeper kafka


echo "Waiting for Kafka container to report healthy status..."
RETRY=0
MAX_RETRIES=60
while :; do
	STATUS=$(docker inspect --format='{{.State.Health.Status}}' $(docker compose ps -q kafka) 2>/dev/null || true)
	if [ "$STATUS" = "healthy" ]; then
		echo "Kafka container is healthy."
		break
	fi
	if [ "$STATUS" = "unhealthy" ]; then
		echo "Kafka reported unhealthy. Showing last kafka logs for debugging:" >&2
		docker compose logs --no-color --tail=200 kafka || true
		exit 1
	fi
	sleep 1
	RETRY=$((RETRY+1))
	if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
		echo "Timed out waiting for Kafka to become healthy after ${MAX_RETRIES} seconds." >&2
		docker compose logs --no-color --tail=200 kafka || true
		exit 1
	fi
done

echo "Kafka is up. Starting Elasticsearch and Kibana..."
docker compose up -d --build elasticsearch kibana

KAFKA_TOPIC_NAME="${KAFKA_TOPIC:-reddit_posts}"
if [ "${KEEP_DATA:-0}" != "1" ]; then
	# Reset Kafka topic to guarantee a fresh pipeline run.
	echo "Resetting Kafka topic ${KAFKA_TOPIC_NAME}..."
	docker compose exec kafka bash -c "\
	kafka-topics --bootstrap-server localhost:9092 --delete --topic ${KAFKA_TOPIC_NAME} >/dev/null 2>&1 || true; \
	kafka-topics --bootstrap-server localhost:9092 --create --topic ${KAFKA_TOPIC_NAME} --partitions 1 --replication-factor 1 >/dev/null 2>&1 || true\
	" || true
	echo "Kafka topic ${KAFKA_TOPIC_NAME} reset."
else
	echo "KEEP_DATA=1 set; skipping Kafka topic reset for ${KAFKA_TOPIC_NAME}."
fi

echo "Waiting for Elasticsearch to become available on http://localhost:9200..."
RETRY=0
MAX_RETRIES=60
while :; do
	STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9200/_cluster/health || true)
	if [ "$STATUS_CODE" = "200" ]; then
		echo "Elasticsearch is available."
		break
	fi
	sleep 1
	RETRY=$((RETRY+1))
	if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
		echo "Timed out waiting for Elasticsearch to become available after ${MAX_RETRIES} seconds." >&2
		docker compose logs --no-color --tail=200 elasticsearch || true
		exit 1
	fi
done

# Ensure reddit_sentiment index exists with the expected mapping so consumers/producers
# can start with a known schema. If it already exists this is a no-op.
ES_INDEX_NAME="${ES_INDEX:-reddit_sentiment}"
if [ "${KEEP_DATA:-0}" != "1" ]; then
	echo "Resetting Elasticsearch index ${ES_INDEX_NAME}..."
	curl -sS -X DELETE "http://localhost:9200/${ES_INDEX_NAME}" >/dev/null 2>&1 || true
	curl -sS -X PUT "http://localhost:9200/${ES_INDEX_NAME}" \
		-H 'Content-Type: application/json' \
		-d '{"mappings":{"properties":{"id":{"type":"keyword"},"created_at":{"type":"date","format":"epoch_second"},"ingested_at":{"type":"date","format":"epoch_second"},"subreddit":{"type":"keyword"},"link_flair_text":{"type":"keyword"},"raw_flair":{"type":"keyword"},"topics":{"type":"keyword"},"title":{"type":"text"},"text":{"type":"text"},"sentiment_label":{"type":"keyword"},"sentiment_score":{"type":"float"}}}}' >/dev/null 2>&1 || true
	echo "Elasticsearch index ${ES_INDEX_NAME} reset."
else
	echo "KEEP_DATA=1 set; skipping Elasticsearch index reset for ${ES_INDEX_NAME}."
fi

echo "Starting es-consumer first (producer will start after model loads)..."
docker compose up -d --build es-consumer

echo ""
echo "⏳ Waiting for sentiment model to load and consumer to connect to Kafka..."
RETRY=0
MAX_RETRIES=120  # 2 minutes for model download + initialization
CONSUMER_READY=false

while [ $RETRY -lt $MAX_RETRIES ]; do
	# Check if consumer loaded the sentiment model
	docker compose logs es-consumer 2>/dev/null | grep -qi "loaded successfully\|Sentiment model loaded\|model loaded" && MODEL_LOADED=1 || MODEL_LOADED=0
	
	# Check if consumer connected to Kafka
	docker compose logs es-consumer 2>/dev/null | grep -qi "Connected to Kafka\|Connecting to Kafka" && KAFKA_CONNECTED=1 || KAFKA_CONNECTED=0
	
	# Check if consumer started the processing loop
	docker compose logs es-consumer 2>/dev/null | grep -qi "Starting Kafka consumer loop\|Kafka consumer loop" && LOOP_STARTED=1 || LOOP_STARTED=0
	
	if [ "$MODEL_LOADED" = "1" ] && [ "$KAFKA_CONNECTED" = "1" ] && [ "$LOOP_STARTED" = "1" ]; then
		echo "✓ Sentiment model loaded!"
		echo "✓ Consumer connected to Kafka!"
		echo "✓ Consumer ready to process messages!"
		CONSUMER_READY=true
		break
	fi
	
	sleep 1
	RETRY=$((RETRY+1))
	if [ $((RETRY % 20)) -eq 0 ]; then
		ELAPSED=$((RETRY / 60))
		echo "  ⏳ Loading... (${ELAPSED}m ${RETRY}s elapsed)"
		docker compose logs es-consumer 2>/dev/null | grep -i "loading\|loaded\|connected" | tail -1
	fi
done

if [ "$CONSUMER_READY" != "true" ]; then
	echo "❌ Consumer failed to load model or connect to Kafka after 2 minutes."
	echo "Last logs:"
	docker compose logs es-consumer 2>/dev/null | tail -10
	exit 1
fi

echo ""
echo "✅ Consumer ready! Now starting reddit-producer..."
docker compose up -d --build reddit-producer

echo ""
echo "Starting search-api..."
docker compose up -d --build search-api

# Check search-api health
echo ""
echo "Checking search-api..."
RETRY=0
MAX_RETRIES=30
while [ $RETRY -lt $MAX_RETRIES ]; do
	HEALTH=$(curl -s --max-time 1 http://localhost:8000/health 2>/dev/null | grep -q '"status":"ok"' && echo "ready" || echo "loading")
	if [ "$HEALTH" = "ready" ]; then
		echo "✓ Search API is ready!"
		break
	fi
	sleep 1
	RETRY=$((RETRY+1))
done

echo ""
echo "=========================================="
echo "✅ System is ready!"
echo "=========================================="
echo ""
echo "Services available at:"
echo "  • Kibana:     http://localhost:5601"
echo "  • Search API: http://localhost:8000/docs"
echo "  • Elasticsearch: http://localhost:9200"
echo ""
echo "To check processing, run:"
echo "  docker compose logs -f es-consumer"
echo ""
echo "Ready to stream Reddit posts with real-time sentiment analysis!"

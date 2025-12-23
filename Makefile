# Convenience Makefile for starting/stopping the Docker Compose pipeline
.PHONY: up down restart logs logs-producer logs-consumer status kibana

up:
	# Start with ordered startup: kafka first, then the rest
	bash scripts/start.sh

# Start without clearing existing Kafka topic / ES index
up-keep:
	KEEP_DATA=1 bash scripts/start.sh

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f

logs-producer:
	docker compose logs -f reddit-producer

logs-consumer:
	docker compose logs -f es-consumer

status:
	docker compose ps

kibana:
	open http://localhost:5601

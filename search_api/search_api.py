#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch
from typing import Optional, List
import uvicorn

load_dotenv()

ES_HOST = os.getenv('ELASTICSEARCH_HOST', 'http://elasticsearch:9200')
ES_INDEX = os.getenv('ES_INDEX', 'reddit_sentiment')

app = FastAPI(title="Reddit Sentiment Search API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ES instance
es = None


@app.on_event("startup")
async def startup_event():
    global es
    print(f"Connecting to Elasticsearch: {ES_HOST}")
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise RuntimeError("Failed to connect to Elasticsearch")
    print("Search API ready")


@app.get("/health")
async def health():
    """Health check endpoint - confirms ES is connected."""
    try:
        if es is None:
            return {"status": "loading"}
        if not es.ping():
            return {"status": "es_disconnected"}
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    subreddit: Optional[str] = Query(None, description="Filter by subreddit"),
    topics: Optional[List[str]] = Query(None, description="Filter by topics"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment: positive, negative, neutral"),
    from_date: Optional[int] = Query(None, description="Start date (epoch seconds)"),
    to_date: Optional[int] = Query(None, description="End date (epoch seconds)"),
    size: int = Query(20, description="Number of results to return", ge=1, le=100)
):
    """
    Text search using BM25 with filtering by topic, sentiment, subreddit, and date range.
    Returns matched posts with sentiment aggregations.
    Multi-word searches boost exact/phrase matches to improve relevance.
    """
    
    # Build filters
    filters = []
    if subreddit:
        filters.append({"term": {"subreddit": subreddit}})
    if topics:
        filters.append({"terms": {"topics": topics}})
    if sentiment:
        filters.append({"term": {"sentiment_label": sentiment.lower()}})
    if from_date or to_date:
        date_filter = {"range": {"created_at": {}}}
        if from_date:
            date_filter["range"]["created_at"]["gte"] = from_date
        if to_date:
            date_filter["range"]["created_at"]["lte"] = to_date
        filters.append(date_filter)
    
    # Build text search query with smarter multi-word handling
    terms = [t for t in q.strip().split() if t]
    is_multi = len(terms) > 1

    # Base must clause (recall-oriented)
    must_clause = {
        "multi_match": {
            "query": q,
            "fields": ["title^2", "text"],
            "type": "best_fields"
        }
    }

    should_clauses: List[dict] = []

    if is_multi:
        # Encourage documents that match all terms across fields
        should_clauses.append({
            "multi_match": {
                "query": q,
                "fields": ["title^2", "text"],
                "type": "cross_fields",
                "operator": "AND",
                "boost": 1.5
            }
        })

        # Strongly boost phrase matches in title and text
        should_clauses.append({
            "match_phrase": {"title": {"query": q, "slop": 1, "boost": 3}}
        })
        should_clauses.append({
            "match_phrase": {"text": {"query": q, "slop": 2, "boost": 2}}
        })

        # Require higher overlap in the base must clause while keeping OR for recall
        must_clause["multi_match"]["operator"] = "OR"
        must_clause["multi_match"]["minimum_should_match"] = "75%"

    query_body = {
        "size": size,
        "query": {
            "bool": {
                "must": [must_clause],
                "filter": filters if filters else [],
                **({"should": should_clauses, "minimum_should_match": 1} if should_clauses else {})
            }
        },
        "_source": [
            "id",
            "title",
            "text",
            "subreddit",
            "topics",
            "raw_flair",
            "created_at",
            "sentiment_label",
            "sentiment_score",
        ],
        "aggs": {
            "sentiment_counts": {
                "terms": {
                    "field": "sentiment_label",
                    "size": 10
                }
            },
            "sentiment_over_time": {
                "date_histogram": {
                    "field": "created_at",
                    "interval": "day",
                    "format": "yyyy-MM-dd"
                },
                "aggs": {
                    "avg_sentiment_score": {
                        "avg": {
                            "field": "sentiment_score"
                        }
                    },
                    "sentiment_breakdown": {
                        "terms": {
                            "field": "sentiment_label"
                        }
                    }
                }
            },
            "top_subreddits": {
                "terms": {
                    "field": "subreddit",
                    "size": 10
                },
                "aggs": {
                    "sentiment_breakdown": {
                        "terms": {
                            "field": "sentiment_label",
                            "size": 3
                        }
                    }
                }
            },
            "top_topics": {
                "terms": {
                    "field": "topics",
                    "size": 10
                },
                "aggs": {
                    "sentiment_breakdown": {
                        "terms": {
                            "field": "sentiment_label",
                            "size": 3
                        }
                    },
                    "avg_sentiment": {
                        "avg": {
                            "field": "sentiment_score"
                        }
                    }
                }
            }
        }
    }
    
    # Execute initial search with strict constraints
    try:
        response = es.search(index=ES_INDEX, body=query_body)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    # If results are low (< 5 hits), retry with more lenient constraints
    initial_hit_count = response['hits']['total']['value']
    if initial_hit_count < 5 and is_multi:
        # Relax multi-word constraints for better recall
        lenient_query_body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": q,
                                "fields": ["title^2", "text"],
                                "type": "best_fields",
                                "operator": "OR"
                            }
                        }
                    ],
                    "filter": filters if filters else [],
                    "should": [
                        {
                            "match_phrase": {"title": {"query": q, "slop": 2, "boost": 2}}
                        },
                        {
                            "match_phrase": {"text": {"query": q, "slop": 3, "boost": 1.5}}
                        }
                    ]
                }
            },
            "_source": [
                "id",
                "title",
                "text",
                "subreddit",
                "topics",
                "raw_flair",
                "created_at",
                "sentiment_label",
                "sentiment_score",
            ],
            "aggs": {
                "sentiment_counts": {
                    "terms": {
                        "field": "sentiment_label",
                        "size": 10
                    }
                },
                "sentiment_over_time": {
                    "date_histogram": {
                        "field": "created_at",
                        "interval": "day",
                        "format": "yyyy-MM-dd"
                    },
                    "aggs": {
                        "avg_sentiment_score": {
                            "avg": {
                                "field": "sentiment_score"
                            }
                        },
                        "sentiment_breakdown": {
                            "terms": {
                                "field": "sentiment_label"
                            }
                        }
                    }
                },
                "top_subreddits": {
                    "terms": {
                        "field": "subreddit",
                        "size": 10
                    },
                    "aggs": {
                        "sentiment_breakdown": {
                            "terms": {
                                "field": "sentiment_label",
                                "size": 3
                            }
                        }
                    }
                },
                "top_topics": {
                    "terms": {
                        "field": "topics",
                        "size": 10
                    },
                    "aggs": {
                        "sentiment_breakdown": {
                            "terms": {
                                "field": "sentiment_label",
                                "size": 3
                            }
                        },
                        "avg_sentiment": {
                            "avg": {
                                "field": "sentiment_score"
                            }
                        }
                    }
                }
            }
        }
        try:
            response = es.search(index=ES_INDEX, body=lenient_query_body)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    
    # Format response
    hits = []
    for hit in response['hits']['hits']:
        source = hit['_source']
        hits.append({
            "id": source.get('id'),
            "title": source.get('title'),
            "text": source.get('text', '')[:300] + ('...' if len(source.get('text', '')) > 300 else ''),
            "subreddit": source.get('subreddit'),
            "topics": source.get('topics', []),
            "raw_flair": source.get('raw_flair'),
            "created_at": source.get('created_at'),
            "sentiment_label": source.get('sentiment_label'),
            "sentiment_score": source.get('sentiment_score'),
            "score": hit['_score']
        })
    
    # Calculate average sentiment as a value between -1 and +1
    sentiment_values = []
    for hit in response['hits']['hits']:
        label = hit['_source'].get('sentiment_label', 'neutral')
        if label == 'positive':
            sentiment_values.append(1)
        elif label == 'negative':
            sentiment_values.append(-1)
        else:  # neutral
            sentiment_values.append(0)
    
    avg_sentiment = sum(sentiment_values) / len(sentiment_values) if sentiment_values else 0
    
    # Format aggregations
    aggs = response.get('aggregations', {})
    sentiment_summary = {
        "total_hits": response['hits']['total']['value'],
        "avg_sentiment": avg_sentiment,
        "sentiment_distribution": {
            bucket['key']: bucket['doc_count'] 
            for bucket in aggs.get('sentiment_counts', {}).get('buckets', [])
        },
        "sentiment_over_time": [
            {
                "date": bucket['key_as_string'],
                "count": bucket['doc_count'],
                "avg_sentiment_score": bucket.get('avg_sentiment_score', {}).get('value'),
                "breakdown": {
                    b['key']: b['doc_count']
                    for b in bucket.get('sentiment_breakdown', {}).get('buckets', [])
                }
            }
            for bucket in aggs.get('sentiment_over_time', {}).get('buckets', [])
        ],
        "top_subreddits": [
            {
                "subreddit": b['key'],
                "count": b['doc_count'],
                "sentiment": {
                    sent['key']: sent['doc_count']
                    for sent in b.get('sentiment_breakdown', {}).get('buckets', [])
                }
            }
            for b in aggs.get('top_subreddits', {}).get('buckets', [])
        ],
        "top_topics": [
            {
                "topic": b['key'],
                "count": b['doc_count'],
                "avg_sentiment": b.get('avg_sentiment', {}).get('value', 0),
                "sentiment": {
                    sent['key']: sent['doc_count']
                    for sent in b.get('sentiment_breakdown', {}).get('buckets', [])
                }
            }
            for b in aggs.get('top_topics', {}).get('buckets', [])
        ]
    }
    
    return {
        "query": q,
        "hits": hits,
        "sentiment_summary": sentiment_summary
    }


@app.get("/stats")
async def stats():
    """Get real-time statistics about indexed documents"""
    try:
        count_result = es.count(index=ES_INDEX)
        return {
            "total_documents": count_result['count'],
            "index": ES_INDEX
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
async def root():
    """Serve the dashboard with embedded global topics data"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if not frontend_path.exists():
        return {"message": "Dashboard not available"}
    
    # Fetch global topics data
    try:
        query_body = {
            "size": 0,
            "query": {"match_all": {}},
            "aggs": {
                "all_topics_with_sentiment": {
                    "terms": {"field": "topics", "size": 50},
                    "aggs": {
                        "sentiment_breakdown": {
                            "terms": {"field": "sentiment_label", "size": 3}
                        },
                        "avg_sentiment_score": {
                            "avg": {"field": "sentiment_score"}
                        }
                    }
                },
                "topics_by_sentiment": {
                    "terms": {"field": "sentiment_label", "size": 3},
                    "aggs": {
                        "top_topics_in_sentiment": {
                            "terms": {"field": "topics", "size": 30},
                            "aggs": {
                                "avg_sentiment_score": {
                                    "avg": {"field": "sentiment_score"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        response = es.search(index=ES_INDEX, body=query_body)
        aggs = response.get("aggregations", {})
        
        # Format global topics data
        all_topics = {}
        for bucket in aggs.get("all_topics_with_sentiment", {}).get("buckets", []):
            topic = bucket["key"]
            sentiment_dist = {
                s["key"]: s["doc_count"]
                for s in bucket.get("sentiment_breakdown", {}).get("buckets", [])
            }
            avg_score = bucket.get("avg_sentiment_score", {}).get("value", 0)
            dominant = max(sentiment_dist, key=sentiment_dist.get) if sentiment_dist else "neutral"
            
            all_topics[topic] = {
                "count": bucket["doc_count"],
                "avg_sentiment_score": avg_score,
                "sentiment_distribution": sentiment_dist,
                "dominant_sentiment": dominant
            }
        
        topics_by_sentiment_breakdown = {}
        for sentiment_bucket in aggs.get("topics_by_sentiment", {}).get("buckets", []):
            sentiment = sentiment_bucket["key"]
            topics_in_sentiment = [
                {
                    "topic": t["key"],
                    "count": t["doc_count"],
                    "avg_sentiment_score": t.get("avg_sentiment_score", {}).get("value", 0)
                }
                for t in sentiment_bucket.get("top_topics_in_sentiment", {}).get("buckets", [])
            ]
            topics_by_sentiment_breakdown[sentiment] = topics_in_sentiment
        
        global_topics_data = {
            "all_topics_with_sentiment": all_topics,
            "topics_by_sentiment": topics_by_sentiment_breakdown
        }
    except Exception as e:
        print(f"Error fetching global topics: {e}")
        global_topics_data = {
            "all_topics_with_sentiment": {},
            "topics_by_sentiment": {}
        }
    
    # Read and modify HTML to embed data
    import json
    html_content = frontend_path.read_text()
    
    # Inject global topics data as inline script
    data_script = f"""
    <script>
        window.GLOBAL_TOPICS_DATA = {json.dumps(global_topics_data)};
    </script>
    """
    
    # Insert before closing </head> tag
    html_content = html_content.replace("</head>", f"{data_script}</head>")
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


@app.get("/topics")
async def topics_page():
    """Serve the dedicated topics overview page"""
    topics_path = Path(__file__).parent.parent / "frontend" / "topics.html"
    if topics_path.exists():
        return FileResponse(topics_path, media_type="text/html")
    return {"message": "Topics page not available"}


@app.get("/top-topics-by-sentiment")
async def top_topics_by_sentiment():
    """Get top topics with granular sentiment breakdown (global, independent of search query)"""
    try:
        query_body = {
            "size": 0,
            "query": {"match_all": {}},
            "aggs": {
                "all_topics_with_sentiment": {
                    "terms": {"field": "topics", "size": 20},
                    "aggs": {
                        "sentiment_breakdown": {
                            "terms": {"field": "sentiment_label", "size": 3}
                        },
                        "avg_sentiment_score": {
                            "avg": {"field": "sentiment_score"}
                        }
                    }
                },
                "topics_by_sentiment": {
                    "terms": {"field": "sentiment_label", "size": 3},
                    "aggs": {
                        "top_topics_in_sentiment": {
                            "terms": {"field": "topics", "size": 10},
                            "aggs": {
                                "avg_sentiment_score": {
                                    "avg": {"field": "sentiment_score"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        response = es.search(index=ES_INDEX, body=query_body)
        aggs = response.get("aggregations", {})
        
        # Format: top topics with sentiment breakdown
        all_topics = {}
        for bucket in aggs.get("all_topics_with_sentiment", {}).get("buckets", []):
            topic = bucket["key"]
            sentiment_dist = {
                s["key"]: s["doc_count"]
                for s in bucket.get("sentiment_breakdown", {}).get("buckets", [])
            }
            avg_score = bucket.get("avg_sentiment_score", {}).get("value", 0)
            dominant = max(sentiment_dist, key=sentiment_dist.get) if sentiment_dist else "neutral"
            
            all_topics[topic] = {
                "count": bucket["doc_count"],
                "avg_sentiment_score": avg_score,
                "sentiment_distribution": sentiment_dist,
                "dominant_sentiment": dominant
            }
        
        # Format: sentiment-first breakdown
        topics_by_sentiment_breakdown = {}
        for sentiment_bucket in aggs.get("topics_by_sentiment", {}).get("buckets", []):
            sentiment = sentiment_bucket["key"]
            topics_in_sentiment = [
                {
                    "topic": t["key"],
                    "count": t["doc_count"],
                    "avg_sentiment_score": t.get("avg_sentiment_score", {}).get("value", 0)
                }
                for t in sentiment_bucket.get("top_topics_in_sentiment", {}).get("buckets", [])
            ]
            topics_by_sentiment_breakdown[sentiment] = topics_in_sentiment
        
        return {
            "all_topics_with_sentiment": all_topics,
            "topics_by_sentiment": topics_by_sentiment_breakdown
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/documents-by-topic")
async def documents_by_topic(
    topic: str = Query(..., description="Topic to filter documents by"),
    sentiment: Optional[str] = Query(None, description="Optional sentiment filter: positive, negative, neutral"),
    size: int = Query(20, description="Number of results", ge=1, le=100),
    offset: int = Query(0, description="Pagination offset", ge=0)
):
    """Return documents filtered by a specific topic and optional sentiment.
    This endpoint does not require a text query and is designed for topic card clicks.
    """
    try:
        filters: List[dict] = [
            {"term": {"topics": topic}}
        ]
        if sentiment:
            filters.append({"term": {"sentiment_label": sentiment.lower()}})

        body = {
            "from": offset,
            "size": size,
            "query": {
                "bool": {
                    "filter": filters
                }
            },
            "_source": [
                "id",
                "title",
                "text",
                "subreddit",
                "topics",
                "raw_flair",
                "created_at",
                "sentiment_label",
                "sentiment_score",
            ],
            "sort": [
                {"created_at": {"order": "desc"}}
            ]
        }

        resp = es.search(index=ES_INDEX, body=body)
        hits = [
            {
                "id": h["_source"].get("id"),
                "title": h["_source"].get("title"),
                "text": h["_source"].get("text", "")[:300] + ("..." if len(h["_source"].get("text", "")) > 300 else ""),
                "subreddit": h["_source"].get("subreddit"),
                "topics": h["_source"].get("topics", []),
                "raw_flair": h["_source"].get("raw_flair"),
                "created_at": h["_source"].get("created_at"),
                "sentiment_label": h["_source"].get("sentiment_label"),
                "sentiment_score": h["_source"].get("sentiment_score"),
                "score": h.get("_score")
            }
            for h in resp.get("hits", {}).get("hits", [])
        ]

        return {
            "topic": topic,
            "sentiment": sentiment or "all",
            "total": resp.get("hits", {}).get("total", {}).get("value", 0),
            "hits": hits
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

#!/usr/bin/env python3
import os
import json
from dotenv import load_dotenv
import requests

load_dotenv()

ES_HOST = os.getenv('ELASTICSEARCH_HOST', 'http://elasticsearch:9200')
ES_INDEX = os.getenv('ES_INDEX', 'reddit_sentiment')
ZERO_SHOT_LABELS = os.getenv('ZERO_SHOT_LABELS', 'politics,technology,sports,art,entertainment,science,health,finance,gaming')
MODEL = os.getenv('ZERO_SHOT_MODEL', 'typeform/distilbert-base-uncased-mnli')

LABELS = [l.strip() for l in ZERO_SHOT_LABELS.split(',') if l.strip()]

def fetch_docs(size=50):
    url = f"{ES_HOST}/{ES_INDEX}/_search?size={size}&sort=created_at:desc"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    hits = data.get('hits', {}).get('hits', [])
    docs = []
    for h in hits:
        src = h.get('_source', {})
        title = src.get('title') or ''
        text = src.get('text') or ''
        combined = f"{title} {text}".strip()
        docs.append({'id': src.get('id'), 'text': combined, 'raw': src})
    return docs

def run_zero_shot(docs):
    from transformers import pipeline
    classifier = pipeline('zero-shot-classification', model=MODEL)
    results = []
    for d in docs:
        if not d['text']:
            results.append({'id': d['id'], 'label': None, 'score': None})
            continue
        try:
            out = classifier(d['text'][:512], LABELS)
            label = out['labels'][0]
            score = out['scores'][0]
            results.append({'id': d['id'], 'label': label, 'score': score, 'all': out})
        except Exception as e:
            results.append({'id': d['id'], 'label': None, 'score': None, 'error': str(e)})
    return results

def main():
    print(f'Fetching up to 50 docs from {ES_HOST}/{ES_INDEX}...')
    docs = fetch_docs(50)
    print(f'Got {len(docs)} docs; running zero-shot with model {MODEL} on labels: {LABELS}')
    res = run_zero_shot(docs)
    counts = {}
    for r in res:
        lbl = r.get('label') or 'none'
        counts[lbl] = counts.get(lbl, 0) + 1
    print('\nLabel counts:')
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')
    print('\nSample results:')
    for r in res[:10]:
        print(json.dumps(r, indent=2))

if __name__ == '__main__':
    main()

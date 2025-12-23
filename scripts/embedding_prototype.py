#!/usr/bin/env python3
import os
import json
from dotenv import load_dotenv
import requests

load_dotenv()

ES_HOST = os.getenv('ELASTICSEARCH_HOST', 'http://elasticsearch:9200')
ES_INDEX = os.getenv('ES_INDEX', 'reddit_sentiment')
LABELS_ENV = os.getenv('ZERO_SHOT_LABELS', '')
DEFAULT_LABELS = 'politics,technology,sports,art,entertainment,science,health,finance,gaming'
LABELS = [l.strip() for l in (LABELS_ENV or DEFAULT_LABELS).split(',') if l.strip()]
EMB_MODEL = os.getenv('EMB_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
FETCH_SIZE = int(os.getenv('EMB_FETCH_SIZE', '50'))


def fetch_docs(size=200):
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


def embed_texts(texts, model_name=EMB_MODEL, batch_size=32):
    # Use transformers to compute mean-pooled embeddings so we don't require sentence-transformers package
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last = out.last_hidden_state  # (batch, seq_len, dim)
            mask = enc['attention_mask'].unsqueeze(-1)
            summed = (last * mask).sum(1)
            denom = mask.sum(1).clamp(min=1e-9)
            mean_pooled = summed / denom
            for vec in mean_pooled.cpu().numpy():
                embeddings.append(vec.tolist())
    return embeddings


def cosine(a, b):
    import math
    sa = sum(x*x for x in a)
    sb = sum(x*x for x in b)
    if sa == 0 or sb == 0:
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (math.sqrt(sa) * math.sqrt(sb))


def main():
    print(f'Fetching docs from {ES_HOST}/{ES_INDEX}...')
    docs = fetch_docs(FETCH_SIZE)
    texts = [d['text'] or '' for d in docs]
    if not texts:
        print('No docs found')
        return
    print(f'Computing embeddings for {len(texts)} docs using {EMB_MODEL}...')
    doc_embs = embed_texts(texts)

    print(f'Computing embeddings for {len(LABELS)} label prototypes: {LABELS}')
    label_embs = embed_texts(LABELS)

    # match
    counts = {}
    top_results = []
    for d, emb in zip(docs, doc_embs):
        best_label = None
        best_score = -1.0
        for lbl, lemb in zip(LABELS, label_embs):
            s = cosine(emb, lemb)
            if s > best_score:
                best_score = s
                best_label = lbl
        counts[best_label] = counts.get(best_label, 0) + 1
        top_results.append({'id': d['id'], 'label': best_label, 'score': best_score, 'text': d['text'][:200]})

    print('\nLabel counts (by nearest prototype):')
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')

    print('\nTop 10 sample matches:')
    for r in top_results[:10]:
        print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()

"""Select distinct-article, human-written SQuAD questions for isolated evaluation."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    raw = Path(args.data).read_bytes()
    data = json.loads(raw)
    rng = random.Random(args.seed)
    articles = []
    for article in data['data']:
        candidates = []
        for para in article['paragraphs']:
            if not 300 <= len(para['context']) <= 2000:
                continue
            for question in para['qas']:
                if not question.get('is_impossible') and 10 <= len(question['question']) <= 200:
                    candidates.append((para['context'], question))
        if candidates:
            articles.append((article['title'], candidates))
    if args.count < 1 or len(articles) < args.count:
        raise ValueError('Not enough distinct articles')
    rng.shuffle(articles)
    styles = ['Answer in one sentence.', 'Answer in approximately 100 words.',
              'Answer in approximately 250 words, explaining the evidence.']
    rows = []
    for i, (title, candidates) in enumerate(articles[:args.count]):
        context, q = rng.choice(candidates)
        prompt = f"Use the passage to answer the question. {styles[i % len(styles)]}\n\nPassage: {context}\n\nQuestion: {q['question']}"
        rows.append({'id': q['id'], 'article': title, 'style': i % len(styles), 'prompt': prompt})
    result = {'source': 'SQuAD v2 development set; human-written English reading-comprehension questions',
              'url': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
              'license': 'CC BY-SA 4.0', 'data_sha256': hashlib.sha256(raw).hexdigest(),
              'seed': args.seed, 'count': len(rows), 'sampling': 'one answerable question per distinct article; artificial answer-style instructions',
              'questions': rows}
    path = Path(args.output)
    if path.exists():
        raise FileExistsError('Refusing to overwrite evaluation manifest')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'questions'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Train a 20-category prompt classifier using labeled synthetic questions."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Labeled questions JSONL, including prompt/category/domain')
    parser.add_argument('--output', default='category_model.joblib')
    parser.add_argument('--report', default='category_report.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = []
    with open(args.data, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('prompt', 'category', 'domain')):
                raise ValueError(f'Invalid prompt/category/domain at line {line_no}')
            rows.append(row)
    texts = np.array([row['prompt'][:1000] for row in rows])
    labels = np.array([row['category'] for row in rows])
    groups = np.array([row['domain'] for row in rows])
    if len(set(groups)) < 3 or len(set(labels)) < 2:
        raise ValueError('Need at least 3 domains and 2 categories')
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(texts, labels, groups))
    if set(labels[train]) != set(labels):
        raise ValueError('Some categories are absent in training domains')
    model = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=3,
                                          max_features=150000, sublinear_tf=True, dtype=np.float32),
                          SGDClassifier(loss='log_loss', random_state=args.seed, max_iter=30, tol=1e-3))
    model.fit(texts[train], labels[train])
    pred = model.predict(texts[test])
    report = {'data': 'synthetic template-generated questions', 'seed': args.seed,
              'split': 'held-out domains; generator templates can overlap',
              'train_count': len(train), 'test_count': len(test),
              'train_domains': sorted(set(groups[train])), 'test_domains': sorted(set(groups[test])),
              'accuracy': float(accuracy_score(labels[test], pred)),
              'macro_f1': float(f1_score(labels[test], pred, average='macro', zero_division=0)),
              'per_category': classification_report(labels[test], pred, output_dict=True, zero_division=0)}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    model.fit(texts, labels)
    joblib.dump({'estimator': model, 'categories': sorted(set(labels)), 'max_chars': 1000,
                 'sklearn_version': sklearn.__version__, 'data_description': report['data']}, args.output)
    print(json.dumps({k: report[k] for k in ('train_count', 'test_count', 'test_domains', 'accuracy', 'macro_f1')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

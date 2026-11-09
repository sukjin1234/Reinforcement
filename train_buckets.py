"""Train and evaluate output-length bucket predictors on held-out domains."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BOUNDS = [256, 768, 1536]
LABELS = ['0-255', '256-767', '768-1535', '1536+']
TEST_DOMAINS = {'경제·경영', '학교·교육'}
VALID_DOMAINS = {'공공·환경'}


def bucket(value):
    return int(np.searchsorted(BOUNDS, value, side='right'))


def score(y, predicted):
    return {'accuracy': float(accuracy_score(y, predicted)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
            'macro_f1': float(f1_score(y, predicted, average='macro', labels=range(4), zero_division=0)),
            'under_bucket_rate': float(np.mean(predicted < y)),
            'confusion_matrix': confusion_matrix(y, predicted, labels=range(4)).tolist()}


def make_model(use_text):
    fields = [('tokens', StandardScaler(with_mean=False), [1]),
              ('category', OneHotEncoder(handle_unknown='ignore'), [2])]
    if use_text:
        fields.append(('text', TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=3,
                                               max_features=100000, dtype=np.float32, sublinear_tf=True), 0))
    return make_pipeline(ColumnTransformer(fields), SGDClassifier(loss='log_loss', random_state=42,
                                                                   max_iter=100, tol=1e-3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', default='bucket_model.joblib')
    parser.add_argument('--report', default='bucket_report.json')
    args = parser.parse_args()
    results = {}
    with open(args.results, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'ok' and r.get('done_reason') == 'stop' and isinstance(r.get('output_tokens'), int) and isinstance(r.get('input_tokens'), int):
                if r['id'] in results:
                    raise ValueError('Duplicate result ID')
                results[r['id']] = r
    data = []
    with open(args.questions, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            if q['id'] in results:
                r = results[q['id']]
                data.append((q['prompt'][:1000], r['input_tokens'], q['category'], q['domain'], bucket(r['output_tokens'])))
    if len(data) < 100:
        raise ValueError('Insufficient valid joined examples')
    x = np.array([[r[0], r[1], r[2]] for r in data], dtype=object)
    y = np.array([r[4] for r in data], dtype=int)
    domains = np.array([r[3] for r in data])
    train = np.array([d not in TEST_DOMAINS | VALID_DOMAINS for d in domains])
    valid = np.array([d in VALID_DOMAINS for d in domains])
    test = np.array([d in TEST_DOMAINS for d in domains])
    if any(set(y[m]) != set(range(4)) for m in (train, valid, test)):
        raise ValueError('Each split must contain all length buckets')
    candidates = {'majority': DummyClassifier(strategy='most_frequent'),
                  'tokens_category': make_model(False), 'text_tokens_category': make_model(True)}
    valid_scores = {}
    for name, model in candidates.items():
        model.fit(x[train], y[train])
        valid_scores[name] = score(y[valid], model.predict(x[valid]))
    chosen = max(candidates, key=lambda name: valid_scores[name]['accuracy'])
    test_scores = {}
    for name, model in candidates.items():
        model.fit(x[train | valid], y[train | valid])
        test_scores[name] = score(y[test], model.predict(x[test]))
    report = {'bounds': BOUNDS, 'labels': LABELS, 'chosen_by_validation_accuracy': chosen,
              'rows': {'train': int(train.sum()), 'validation': int(valid.sum()), 'test': int(test.sum())},
              'validation_domains': sorted(VALID_DOMAINS), 'test_domains': sorted(TEST_DOMAINS),
              'validation': valid_scores, 'test': test_scores,
              'limitations': 'Generated synthetic templates can overlap across domains. True category is used; real-world generalization and automatic category errors are not measured. Outputs stopped by length are excluded.'}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    model = candidates[chosen]
    model.fit(x, y)
    joblib.dump({'estimator': model, 'bounds': BOUNDS, 'labels': LABELS,
                 'max_chars': 1000, 'selection': chosen}, args.output)
    print(json.dumps({'rows': report['rows'], 'selection': chosen, 'test': test_scores}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Classify a user prompt; model scores are not calibrated probabilities."""
import math


def classify(bundle, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('A nonempty text user message is required for automatic classification')
    model = bundle['estimator']
    sample = text[:bundle['max_chars']]
    scores = model.predict_proba([sample])[0]
    index = int(scores.argmax())
    score = float(scores[index])
    if not math.isfinite(score):
        raise ValueError('Invalid category score')
    return {'category': str(model.classes_[index]), 'score': score,
            'truncated': len(text) > bundle['max_chars'],
            'notice': 'Synthetic-data classifier; verify category or supply it manually. Score is not calibrated.'}

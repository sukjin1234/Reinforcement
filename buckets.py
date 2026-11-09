"""Shadow predictor for output-length ranges."""
import math


def predict_bucket(bundle, text, input_tokens, category):
    if not isinstance(text, str) or not text.strip():
        return None
    model = bundle['estimator']
    scores = model.predict_proba([[text[:bundle['max_chars']], input_tokens, category]])[0]
    index = int(scores.argmax())
    label_index = int(model.classes_[index])
    value = float(scores[index])
    if not math.isfinite(value):
        raise ValueError('Invalid bucket score')
    return {'bucket': bundle['labels'][label_index], 'score': value,
            'text_truncated': len(text) > bundle['max_chars'],
            'notice': 'Experimental synthetic-data model; score is not calibrated.'}

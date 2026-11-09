"""Per-request work estimates; vLLM engine queue metrics are collected separately."""
import math
import time


def estimate_seconds(input_tokens, output_tokens, performance):
    in_rate = performance.get('uncached_input_tokens_per_second')
    out_rate = performance.get('output_tokens_per_second')
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (in_rate, out_rate)):
        return None
    return input_tokens / in_rate + output_tokens / out_rate


def snapshot(work, now=None):
    elapsed = max(0, (time.monotonic() if now is None else now) - work['start'])
    total = work['estimated_service_seconds']
    return {'input_tokens': work['input_tokens'], 'predicted_output_tokens': work['predicted_output_tokens'],
            'elapsed_seconds': elapsed, 'estimated_service_seconds': total,
            'estimated_remaining_seconds': max(0, total - elapsed) if total is not None else None,
            'overdue': elapsed > total if total is not None else None}

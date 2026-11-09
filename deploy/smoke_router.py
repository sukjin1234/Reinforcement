"""Exercise the router without supplying input token counts."""
import json
import sys
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8009'
body = {'messages': [{'role': 'system', 'content': 'Answer in one short sentence.'},
                     {'role': 'user', 'content': 'What is a GPU?'}]}


def post(path):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=180)


with post('/chat') as response:
    result = json.load(response)
    assert result['token_count_matches'], result
    print(json.dumps({k: result[k] for k in ('server_id', 'input_tokens', 'actual_prompt_tokens', 'category', 'predicted_output_tokens', 'token_count_matches')}), flush=True)
with post('/chat/stream') as response:
    events = [json.loads(line) for line in response if line.strip()]
    complete = events[-1]
    assert complete['event'] == 'complete' and complete['token_count_matches'], complete
    print(json.dumps({k: complete[k] for k in ('server_id', 'actual_prompt_tokens', 'actual_output_tokens', 'token_count_matches', 'ttft_seconds')}), flush=True)

"""Check actual chat token counts against the same vLLM server's usage."""
import json
import sys
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8100'
cases = [
    [{'role': 'user', 'content': '안녕하세요. 한 문장으로 답변해 주세요.'}],
    [{'role': 'system', 'content': 'Answer briefly.'}, {'role': 'user', 'content': 'What is a GPU?'}],
    [{'role': 'user', 'content': 'What is 1+1?'}, {'role': 'assistant', 'content': '2'},
     {'role': 'user', 'content': '그리고 2+2는?'}],
]


def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


for messages in cases:
    common = {'model': 'qwen3-8b', 'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}}
    tokenized = post('/tokenize', common)['count']
    usage = post('/v1/chat/completions', {**common, 'max_tokens': 16, 'temperature': 0})['usage']
    print(json.dumps({'tokenized': tokenized, 'prompt_tokens': usage['prompt_tokens'],
                      'output_tokens': usage['completion_tokens'], 'matches': tokenized == usage['prompt_tokens']}), flush=True)
    if tokenized != usage['prompt_tokens']:
        raise SystemExit('Token count mismatch')

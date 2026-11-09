"""Extend only the existing round-tunnel restricted key to local vLLM."""
from pathlib import Path

path = Path.home() / '.ssh' / 'authorized_keys'
lines = path.read_text().splitlines(keepends=True)
matches = [i for i, line in enumerate(lines) if line.rstrip().endswith(' round-tunnel')]
if len(matches) != 1:
    raise SystemExit('Expected exactly one round-tunnel key; no changes made')
i = matches[0]
before = 'permitopen="127.0.0.1:11434"'
after = before + ',permitopen="127.0.0.1:8100"'
if after not in lines[i]:
    if before not in lines[i] or 'restrict,port-forwarding,' not in lines[i]:
        raise SystemExit('Unexpected key restrictions; no changes made')
    lines[i] = lines[i].replace(before, after, 1)
    path.write_text(''.join(lines))

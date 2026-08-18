with open('/Users/brian/air/Apex_oracle_bot/src/circuit_breaker.py', 'rb') as f:
    content = f.read()
lines = content.split(b'\n')
for i, line in enumerate(lines[73:82], 76):
    has_tab = line.startswith(b'\t')
    print(f'Line {i+76}: len={len(line)}, starts_space={line.startswith(b" ")}, has_tab={line.startswith(b"\t")}, repr={repr(line[:80])}')
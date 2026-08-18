with open('/Users/brian/air/Apex_oracle_bot/src/circuit_breaker.py', 'rb') as f:
    content = f.read()
lines = content.split(b'\n')
for i, line in enumerate(lines[74:82], 76):
    indent = len(line) - len(line.lstrip(b' '))
    has_tab = line.startswith(b'\t')
    print(f'Line {i+76}: len={len(line)}, indent={indent}, has_tab={has_tab}, repr={repr(line[:80])}')
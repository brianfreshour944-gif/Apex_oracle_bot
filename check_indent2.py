with open('/Users/brian/air/Apex_oracle_bot/src/circuit_breaker.py', 'rb') as f:
    content = f.read()
lines = content.split(b'\n')
for i, line in enumerate(lines[73:82], 76):
    has_tab = line.startswith(b'\t')
    has_weird = any(b in line for b in [0x0b, 0x0c, 0x0d, 0x85, 0xA0, 0x2000, 0x200B, 0xFEFF])
    print(f'Line {i+76}: len={len(line)}, starts_with_space={line.startswith(b" ")}, has_tab={line.startswith(b"\\t")}, has_weird={any(b in line for b in [0x0b, 0x0c, 0x0d, 0x85, 0xA0, 0x2000, 0x200B, 0xFEFF])}, repr={repr(line[:80])}')
# Fix bot.py - fix blank line indentations
with open('src/bot.py', 'r') as f:
    lines = f.readlines()

# Check current state
print(f"Line 150 (index 149): len={len(lines[149])}, repr={repr(lines[149])}")
print(f"Line 153 (index 152): len={len(lines[152])}, repr={repr(lines[152])}")

# Fix blank lines - write exactly 29 spaces + newline
lines[149] = ' ' * 29 + '\n'  # line 150
lines[152] = ' ' * 29 + '\n'  # line 153

with open('src/bot.py', 'w') as f:
    f.writelines(lines)

print("Fixed blank line indentations")

# Verify
with open('src/bot.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[145:170], 146):
    indent = len(line) - len(line.lstrip())
    print(f'{i}: indent={indent} {repr(line[:80])}')
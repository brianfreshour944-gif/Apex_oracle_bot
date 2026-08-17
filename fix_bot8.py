# Fix bot.py - fix blank line indentations
with open('src/bot.py', 'r') as f:
    lines = f.readlines()

# Fix blank lines at indices 149 and 152 (lines 150 and 153) - should be 29 spaces
lines[149] = '                             \n'  # line 150
lines[152] = '                             \n'  # line 153

with open('src/bot.py', 'w') as f:
    f.writelines(lines)

print("Fixed blank line indentations")

# Verify
with open('src/bot.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[145:170], 146):
    indent = len(line) - len(line.lstrip())
    print(f'{i}: indent={indent} {repr(line[:80])}')
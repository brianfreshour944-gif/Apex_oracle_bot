# Fix bot.py - fix the blank line indentation
with open('src/bot.py', 'r') as f:
    lines = f.readlines()

# Fix line 150 (index 149) - blank line should be 29 spaces
lines[149] = '                             \n'

with open('src/bot.py', 'w') as f:
    f.writelines(lines)

print("Fixed blank line indentation")

# Verify
with open('src/bot.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[145:165], 146):
    indent = len(line) - len(line.lstrip())
    print(f'{i}: indent={indent} {repr(line[:80])}')
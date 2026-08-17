# Fix bot.py - force write with correct indentation
with open('src/bot.py', 'r') as f:
    content = f.read()

# Replace the blank lines with correct indentation
# Line 150: 30 spaces -> 29 spaces
# Line 153: 30 spaces -> 29 spaces
content = content.replace('                             \n', '                             \n', 2)  # This won't work since both have same pattern

# Better approach: do line-by-line
lines = content.splitlines(keepends=True)
lines[149] = ' ' * 29 + '\n'
lines[152] = ' ' * 29 + '\n'

with open('src/bot.py', 'w') as f:
    f.write(''.join(lines))

print("Fixed blank line indentations")

# Verify
with open('src/bot.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[145:170], 146):
    indent = len(line) - len(line.lstrip())
    print(f'{i}: indent={indent} {repr(line[:80])}')
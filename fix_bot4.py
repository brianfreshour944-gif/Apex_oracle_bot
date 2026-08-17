# Fix bot.py - fix the indentation to be 29 spaces (4 more than if at 25)
with open('src/bot.py', 'r') as f:
    lines = f.readlines()

# Fix line 151 (index 150) - should be 29 spaces
lines[150] = '                             x = torch.tensor(data_scaled).unsqueeze(0).to(device)\n'

# Fix line 153 (index 152) - blank line should be 29 spaces
lines[152] = '                             \n'

with open('src/bot.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation")

# Verify
with open('src/bot.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[145:165], 146):
    indent = len(line) - len(line.lstrip())
    print(f'{i}: indent={indent} {repr(line[:80])}')
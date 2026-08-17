# Fix bot.py - fix line 152 indentation
with open('src/bot.py', 'r') as f:
    lines = f.readlines()

# Fix line 152 (index 151) - should be 29 spaces
lines[151] = '                             label_tensor = torch.tensor([[1.0 if realized_pnl > 0 else 0.0]], dtype=torch.float32).to(device)\n'

with open('src/bot.py', 'w') as f:
    f.writelines(lines)

print("Fixed line 152 indentation")

# Verify
with open('src/bot.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[145:165], 146):
    indent = len(line) - len(line.lstrip())
    print(f'{i}: indent={indent} {repr(line[:80])}')
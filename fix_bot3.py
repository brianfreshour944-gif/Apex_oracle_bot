# Fix bot.py - fix the indentation issue
with open('src/bot.py', 'r') as f:
    lines = f.readlines()

# Fix line 151 (index 150) - x = torch.tensor... should have proper indentation
# Line 151 is index 150 (0-indexed)
# Line 151 currently has indent 0, should be 30 spaces (inside the if block)
# Line 152 (label_tensor) has indent 30, which is correct
# Line 153 (blank) has indent 31, should be 30
# Line 154 (comment) has indent 30, correct
# Line 155 (with lock) has indent 30, correct

# Fix line 151 (index 150)
lines[150] = '                              x = torch.tensor(data_scaled).unsqueeze(0).to(device)\n'

# Fix line 153 (index 152) - blank line should have 30 spaces
lines[152] = '                              \n'

# Also need to check the except block alignment
# The except block at line 186 (index 185) should align with the try at line 127 (index 126)
# Let's check the structure

with open('src/bot.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation")

# Verify
with open('src/bot.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[145:165], 146):
    indent = len(line) - len(line.lstrip())
    print(f'{i}: indent={indent} {repr(line[:80])}')
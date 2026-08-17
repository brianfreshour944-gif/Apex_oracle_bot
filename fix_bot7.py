# Fix bot.py - fix the with statement and comment indentation
with open('src/bot.py', 'r') as f:
    lines = f.readlines()

# Fix lines 154-155 (indices 153-154) - should be 29 spaces (inside if block)
# Line 154 (index 153): comment
lines[153] = '                             # Protect model train/eval state from concurrent a\n'
# Line 155 (index 154): with statement
lines[154] = '                             with _model_inference_lock:\n'

# Also fix the blank line at line 150 (index 149) - should be 29 spaces
lines[149] = '                             \n'

# Also fix line 153 (blank line after label_tensor) - index 152, should be 29 spaces
lines[152] = '                             \n'

with open('src/bot.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation")

# Verify
with open('src/bot.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[145:170], 146):
    indent = len(line) - len(line.lstrip())
    print(f'{i}: indent={indent} {repr(line[:80])}')
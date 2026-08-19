with open('src/bot.py', 'r') as f:
    lines = f.readlines()

# Lines 213-219 (0-indexed 212-218) need to be dedented from inside the else: block to run unconditionally after the if/else.
# Current indentation: 39 spaces (inside else:)
# Target indentation: 34 spaces (same level as the if/else)

# Also fix the nested indentation for the for/if inside with torch.no_grad():
# Line 215 (0-indexed): 'with torch.no_grad():' at 34 spaces
# Line 216: 'for p in model.parameters():' should be at 38 spaces
# Line 217: 'if p.grad is not None:' should be at 42 spaces
# Line 218: 'p.data -= lr * p.grad' should be at 46 spaces

# Fix lines 212-218
lines[212] = ' ' * 34 + '\n'  # blank line
lines[213] = ' ' * 34 + '_transformer_online_updates += 1\n'
lines[214] = ' ' * 34 + 'with torch.no_grad():\n'
lines[215] = ' ' * 38 + 'for p in model.parameters():\n'
lines[216] = ' ' * 42 + 'if p.grad is not None:\n'
lines[217] = ' ' * 46 + 'p.data -= lr * p.grad\n'
lines[218] = ' ' * 34 + 'model.eval()\n'

with open('src/bot.py', 'w') as f:
    f.writelines(lines)
print("FIX applied successfully")
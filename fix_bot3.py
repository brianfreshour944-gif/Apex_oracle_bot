import sys

with open('src/bot.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the index of the line that contains "cooldowns: Dict[str, float] = {}"
cooldowns_idx = -1
for i, line in enumerate(lines):
    if line.strip() == 'cooldowns: Dict[str, float] = {}':
        cooldowns_idx = i
        break

if cooldowns_idx == -1:
    print("cooldowns line not found")
    sys.exit(1)

# Remove any consecutive lines after the cooldowns line that are "_symbol_locks: Dict[str, asyncio.Lock] = {}" (after stripping)
i = cooldowns_idx + 1
while i < len(lines) and lines[i].strip() == '_symbol_lstrip() == '_symbol_locks: Dict[str, asyncio.Lock] = {}':
    # Remove this line
    del lines[i]
    # Do not increment i because we removed the current element

# Now insert the line after the cooldowns line
lines.insert(cooldowns_idx + 1, '_symbol_locks: Dict[str, asyncio.Lock] = {}\n')

with open('src/bot.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
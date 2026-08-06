import sys

with open('src/bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the line
new_content = content.replace('cooldowns: Dict[str, float] = {}\n', 'cooldowns: Dict[str, float] = {}\n_symbol_locks: Dict[str, asyncio.Lock] = {}\n')

with open('src/bot.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
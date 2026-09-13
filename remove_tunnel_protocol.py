with open('/Users/brian/air/Apex_oracle_bot/src/bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

start = content.find('def read_tunnel_status() -> bool:')
end = content.find('async def process_signal_for_symbol')
if start >= 0 and end > start:
    content = content[:start] + content[end:]
    with open('/Users/brian/air/Apex_oracle_bot/src/bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Removed tunnel protocol functions from bot.py')
else:
    print('Tunnel functions not found or already removed')

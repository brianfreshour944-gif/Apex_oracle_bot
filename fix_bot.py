import sys

with open('src/bot.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start index of the function
start_idx = -1
for i, line in enumerate(lines):
    if line.strip().startswith('async def process_signal_for_symbol('):
        start_idx = i
        break

if start_idx == -1:
    print('Function not found')
    sys.exit(1)

# Find the end index: look for the next function definition after start_idx
end_idx = -1
for i in range(start_idx + 1, len(lines)):
    if lines[i].strip().startswith('async def ') and not lines[i].strip().startswith('async def process_signal_for_symbol('):
        end_idx = i
        break

if end_idx == -1:
    # If not found, assume the function goes to the end of the file
    end_idx = len(lines)

# The function is from start_idx to end_idx-1 (because the next function starts at end_idx)
# We want to replace lines[start_idx:end_idx] with the new function lines.

new_func_lines = [
    'async def process_signal_for_symbol(symbol: str, current_price: float, risk_manager: RiskManager, strategy: TradingStrategy, ex: AlpacaExchange, regime_flag: dict = None, banned_symbols: set = None) -> None:\n',
    '    \"\"\"Processes signal for a single symbol asynchronously.\"\"\"\n',
    '    # Get or create lock for this symbol\n',
    '    lock = _symbol_limits.setdefault(symbol, asyncio.Lock())\n',
    '    async with lock:\n',
]

# Now, for each line in the original function from start_idx+2 to end_idx-1, we prepend 4 spaces.
for i in range(start_idx+2, end_idx):
    # If the line is empty, we still add 4 spaces? We'll just add 4 spaces and the line.
    new_func_lines.append('    ' + lines[i])

# Now, replace the lines from start_idx to end_idx-1 with new_func_lines
lines = lines[:start_idx] + new_func_lines + lines[end_idx:]

# Write back to file
with open('src/bot.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
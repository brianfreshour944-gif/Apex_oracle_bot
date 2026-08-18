with open('/Users/brian/air/Apex_oracle_bot/src/feature_engineering.py', 'rb') as f:
    content = f.read()

# Find the problematic area - the try: at function level that lost its indentation
# Find the docstring end and the try: that follows
lines = content.split(b'\n')

# Find the function definition line
func_def_line = -1
for i, line in enumerate(lines):
    if b'async def add_multi_timeframe_features' in line:
        func_def_line = i
        break

print(f"Function definition at line {func_def_line}")

# Find the docstring end
docstring_end = -1
for i in range(func_def_line, len(lines)):
    if b'    """' in lines[i]:
        docstring_end = i
        break

print(f"Docstring ends at line {docstring_end}: {repr(lines[docstring_end])}")

# Find the try: that should be indented
for i in range(docstring_end + 1, min(docstring_end + 20, len(lines))):
    if b'try:' in lines[i]:
        print(f"Line {i}: indent={len(lines[i]) - len(lines[i].lstrip(b' '))} {repr(lines[i])}")

# Find the problematic except
for i in range(len(lines)):
    if b'except Exception as e:' in lines[i]:
        indent = len(lines[i]) - len(lines[i].lstrip(b' '))
        print(f"Line {i}: indent={len(lines[i]) - len(lines[i].lstrip(b' '))} {repr(lines[i][:80])}")

# Find the try: at function level
for i in range(len(lines)):
    if lines[i].strip() == b'try:':
        indent = len(lines[i]) - len(lines[i].lstrip(b' '))
        print(f"Line {i}: indent={len(lines[i]) - len(lines[i].lstrip(b' '))} {repr(lines[i][:80])}")
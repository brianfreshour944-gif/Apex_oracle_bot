import ast, os

ROOT = r"C:\TradingBots\Apex_oracle_bot-main"
unresolved = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    if "__pycache__" in dirpath or ".venv" in dirpath or ".git" in dirpath:
        continue
    for fn in filenames:
        if fn.endswith(".py"):
            p = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
                    f = node.value.func
                    key = None
                    if isinstance(f, ast.Name):
                        key = f.id
                    elif isinstance(f, ast.Attribute):
                        base = f.value.id if isinstance(f.value, ast.Name) else "?"
                        key = f"{base}.{f.attr}"
                    if key and not (key.startswith(("asyncio.", "self._", "self.")) or key in ("gather",)):
                        unresolved.setdefault(key, []).append(os.path.relpath(p, ROOT) + f":{f.lineno}")
for k in sorted(unresolved):
    print(k, "->", unresolved[k][:4])

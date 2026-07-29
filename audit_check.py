"""
Comprehensive async/await audit script.
Checks every await call and whether the function being awaited is actually async.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def get_all_py_files():
    result = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in ('__pycache__', '.git', '.pytest_cache', '.venv', 'venv')]
        for fname in filenames:
            if fname.endswith('.py'):
                result.append(os.path.join(dirpath, fname))
    return result

def collect_async_defs(tree):
    """Collect all async def names from a file's AST."""
    async_funcs = set()
    sync_funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            async_funcs.add(node.name)
        elif isinstance(node, ast.FunctionDef):
            sync_funcs.add(node.name)
    return async_funcs, sync_funcs

def get_call_name(call_node):
    """Extract a human-readable name from a Call node's func."""
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    elif isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}"
        return f"?.{func.attr}"
    return None

def check_file_awaits(filepath):
    """Parse a file and list all await calls."""
    with open(filepath, encoding='utf-8') as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {filepath}: {e}")
        return []

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Await):
            val = node.value
            if isinstance(val, ast.Call):
                name = get_call_name(val)
                if name:
                    issues.append((node.lineno, name))
    return issues

def main():
    files = get_all_py_files()
    print(f"Found {len(files)} Python files\n")

    # First pass: collect all async/sync definitions across all files
    all_async = {}
    all_sync = {}
    all_trees = {}
    for fp in files:
        try:
            with open(fp, encoding='utf-8') as f:
                src = f.read()
            tree = ast.parse(src)
            all_trees[fp] = tree
            af, sf = collect_async_defs(tree)
            for fn in af:
                all_async.setdefault(fn, []).append(fp)
            for fn in sf:
                all_sync.setdefault(fn, []).append(fp)
        except Exception:
            pass

    # Second pass: check awaits
    print("=" * 70)
    print("AWAIT CALL AUDIT")
    print("=" * 70)
    for fp in sorted(files):
        rel = os.path.relpath(fp, ROOT)
        awaits = check_file_awaits(fp)
        if awaits:
            print(f"\n--- {rel} ---")
            for lineno, name in awaits:
                func_name = name.split('.')[-1]  # just the method/function name
                in_async = func_name in all_async
                in_sync = func_name in all_sync
                if in_sync and not in_async:
                    status = "*** BUG: SYNC FUNCTION AWAITED ***"
                elif in_async:
                    status = "OK (async confirmed)"
                else:
                    status = "UNKNOWN (external/library)"
                print(f"  L{lineno:4d}: await {name}()  -> {status}")

    # Check for duplicate/conflicting files
    print("\n\n" + "=" * 70)
    print("DUPLICATE FILE CHECK")
    print("=" * 70)
    basenames = {}
    for fp in files:
        base = os.path.basename(fp)
        basenames.setdefault(base, []).append(fp)
    for base, paths in sorted(basenames.items()):
        if len(paths) > 1:
            print(f"DUPLICATE: {base}")
            for p in paths:
                print(f"  {os.path.relpath(p, ROOT)}")

    # Check imports vs usage for each file
    print("\n\n" + "=" * 70)
    print("IMPORT AUDIT (names used but not imported)")
    print("=" * 70)
    # We'll do a simple grep-style check for the most suspicious patterns

if __name__ == "__main__":
    main()

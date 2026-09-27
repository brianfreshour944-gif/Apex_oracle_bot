import ast
import os

ROOT = r"C:\TradingBots\Apex_oracle_bot-main"


class _AwaitVisitor(ast.NodeVisitor):
    """Walk a module AST reporting `await` targets that resolve to a sync def.

    Per-file context (path/funcs/methods/imports) is carried on the instance
    rather than captured from an enclosing loop, so the visitor always sees
    the file it was constructed for (no late-binding surprise).
    """

    def __init__(self, p, funcs, methods, imports, global_index):
        self.p = p
        self.funcs = funcs
        self.methods = methods
        self.imports = imports
        self.global_index = global_index
        self.findings = []
        self.stack = []

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Await(self, node):
        p, funcs, methods, imports = self.p, self.funcs, self.methods, self.imports
        f = node.value
        target_async = None
        desc = None
        if isinstance(f, ast.Name):
            if f.id in funcs:
                target_async = funcs[f.id]
                desc = f"{f.id} (local def)"
            elif f.id in imports:
                mod, orig = imports[f.id]
                if orig is None:
                    desc = f"{f.id} (module import {mod})"
                    # module attribute await e.g. await mod.func handled below
                else:
                    # imported name; resolve across files
                    cands = global_index.get(orig, set())
                    if cands:
                        target_async = all(a for _, a in cands)
                        desc = f"{orig} imported from {mod} -> defs {[os.path.basename(c) for c, _ in cands]} async={[a for _, a in cands]}"
                    else:
                        desc = f"{orig} imported from {mod} (no py def found)"
            else:
                cands = global_index.get(f.id, set())
                if cands:
                    target_async = all(a for _, a in cands)
                    desc = f"{f.id} -> {[(os.path.relpath(c, ROOT), a) for c, a in cands]}"
        elif isinstance(f, ast.Attribute):
            desc = ast.unparse(f)
            if isinstance(f.value, ast.Name) and f.value.id in imports:
                mod, orig = imports[f.value.id]
                if orig is None:
                    # await module.attr
                    modfile = None
                    for q in files:
                        base = os.path.basename(q)[:-3]
                        if base == mod.split(".")[-1] or q.replace("\\\\", "/").endswith(mod.replace(".", "/") + ".py"):
                            modfile = q
                            break
                    if modfile:
                        other = files[modfile][1]
                        if f.attr in other:
                            target_async = other[f.attr]
                            desc += f" resolved in {os.path.relpath(modfile, ROOT)}"
            elif isinstance(f.value, ast.Name) and self.stack:
                # self.method()
                for cls in self.stack:
                    if f.attr in methods.get(cls, {}):
                        target_async = methods[cls][f.attr]
                        desc += f" (method of {cls})"
        if target_async is False:
            self.findings.append((os.path.relpath(p, ROOT), getattr(f, "lineno", node.lineno), desc))
        self.generic_visit(node)


# pass 1: collect all function defs and their async-ness, per file, plus module-level imports
files = {}
for dirpath, _dirnames, filenames in os.walk(ROOT):
    if "__pycache__" in dirpath or ".venv" in dirpath or ".git" in dirpath:
        continue
    for fn in filenames:
        if fn.endswith(".py"):
            p = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except SyntaxError as e:
                print("SYNTAXERR", p, e)
                continue
            funcs = {}  # name -> is_async (incl methods, class-qualified later)
            methods = {}  # class -> {name: is_async}
            imports = {}  # local name -> (module, orig name)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs[node.name] = isinstance(node, ast.AsyncFunctionDef)
                elif isinstance(node, ast.ClassDef):
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.setdefault(node.name, {})[sub.name] = isinstance(sub, ast.AsyncFunctionDef)
                elif isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        imports[a.asname or a.name] = (node.module, a.name)
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        imports[(a.asname or a.name).split(".")[0]] = (a.name, None)
            files[p] = (tree, funcs, methods, imports)

# global index: function name -> set of (modulepath, is_async)
global_index = {}
for _p, (_tree, funcs, _methods, _imports) in files.items():
    for name, is_a in funcs.items():
        global_index.setdefault(name, set()).add((_p, is_a))

findings = []
for p, (tree, funcs, methods, imports) in files.items():
    visitor = _AwaitVisitor(p, funcs, methods, imports, global_index)
    visitor.visit(tree)
    findings.extend(visitor.findings)

print(f"== {len(findings)} awaited-name resolves-to-sync findings ==")
for rel, line, desc in findings:
    print(f"{rel}:{line}: await -> {desc}")

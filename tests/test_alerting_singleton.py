"""Regression test for a recurring merge-conflict bug: bot.py's alerting
import silently flipping from the singleton getter to the raw class.

History: commits 48c6226 and its merge into 2c37f0d both replaced
`from src.alerting import get_alerting_engine` with
`from src.alerting import AlertingEngine` -- a change that passes static
analysis (both names are real, importable) but crashes at runtime with
NameError, because bot.py CALLS get_alerting_engine() at ~lines 1809/1869,
it never instantiates AlertingEngine() directly. This test parses bot.py's
own source to confirm the correct name is imported and used, so this
specific regression fails a test run instead of a production deploy.
"""
import ast
import re
from pathlib import Path

BOT_PY = Path(__file__).parent.parent / "src" / "bot.py"


def test_bot_imports_get_alerting_engine_not_the_class():
    """bot.py must import the get_alerting_engine() singleton getter.

    Importing AlertingEngine (the class) instead is the recurring regression
    this test guards against -- see module docstring above.
    """
    source = BOT_PY.read_text()
    tree = ast.parse(source, filename=str(BOT_PY))

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.alerting":
            for alias in node.names:
                imported_names.add(alias.name)

    assert "get_alerting_engine" in imported_names, (
        "bot.py must `from src.alerting import get_alerting_engine` -- "
        f"found these src.alerting imports instead: {imported_names or 'NONE'}. "
        "This is the recurring regression from commits 48c6226/2c37f0d: "
        "importing the AlertingEngine class instead of the singleton getter "
        "function causes a NameError at runtime, since bot.py calls "
        "get_alerting_engine() as a function, not AlertingEngine() as a "
        "constructor."
    )


def test_bot_calls_get_alerting_engine_as_function():
    """Sanity check the other half: bot.py must actually CALL the getter
    (not just import it) at its known usage sites, confirming the import
    and usage are consistent with each other.
    """
    source = BOT_PY.read_text()
    assert re.search(r"get_alerting_engine\(\)", source), (
        "bot.py no longer calls get_alerting_engine() anywhere -- if the "
        "alerting integration was intentionally removed or refactored, "
        "update or remove this test accordingly. If not, this may indicate "
        "the alerting engine initialization was accidentally deleted."
    )


def test_bot_py_compiles():
    """Belt-and-suspenders: confirm bot.py is at least syntactically valid
    Python, so a NameError-style regression doesn't slip through with the
    file failing to even parse.
    """
    source = BOT_PY.read_text()
    compile(source, str(BOT_PY), "exec")

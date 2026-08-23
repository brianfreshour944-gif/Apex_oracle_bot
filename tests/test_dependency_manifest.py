"""
pyproject.toml used to list alpaca-trade-api==3.2.0 (the legacy library),
while requirements.txt correctly listed alpaca-py>=0.30.0 -- the actual
package src/exchange.py imports from (`alpaca.trading.*`, `alpaca.data.*`,
which only exist in alpaca-py, not alpaca-trade-api). `pip install -e .`
would have pulled in the wrong package and left every import in
src/exchange.py broken.

These are fast, static checks against the manifest files (no real install)
so a future edit reintroducing the mismatch fails CI immediately; the
review's own suggested verification -- pip install -e . in a real clean
venv, then `python -c "from src.exchange import AlpacaExchange"` -- was run
by hand separately and isn't repeated here since it's slow and needs
network access to PyPI.
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_dependencies() -> list[str]:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["dependencies"]


def _requirements_lines() -> list[str]:
    with open(REPO_ROOT / "requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def test_pyproject_does_not_depend_on_legacy_alpaca_trade_api():
    deps = _pyproject_dependencies()
    assert not any(d.startswith("alpaca-trade-api") for d in deps), (
        "pyproject.toml still lists the legacy alpaca-trade-api package -- "
        "src/exchange.py imports from alpaca.trading.*/alpaca.data.*, which "
        "only exist in alpaca-py."
    )


def test_pyproject_depends_on_alpaca_py():
    deps = _pyproject_dependencies()
    assert any(d.startswith("alpaca-py") for d in deps), (
        "pyproject.toml must list alpaca-py -- it's the package src/exchange.py actually imports."
    )


def test_pyproject_and_requirements_agree_on_alpaca_py():
    """Both manifests should express the same minimum version constraint,
    not just both happen to mention the package name."""
    py_deps = _pyproject_dependencies()
    req_lines = _requirements_lines()

    py_alpaca = next((d for d in py_deps if d.startswith("alpaca-py")), None)
    req_alpaca = next((line for line in req_lines if line.startswith("alpaca-py")), None)

    assert py_alpaca is not None
    assert req_alpaca is not None

    def _min_version(spec: str) -> str:
        m = re.search(r">=\s*([\d.]+)", spec)
        assert m, f"expected a >= version constraint in {spec!r}"
        return m.group(1)

    assert _min_version(py_alpaca) == _min_version(req_alpaca)


def test_pyproject_declares_wheel_package_location():
    """
    Separate bug from the alpaca-py mismatch, discovered while actually
    running `pip install -e .` in a clean venv per this fix's own
    verification step: pyproject.toml's project name (apex-oracle-bot)
    doesn't match any directory at the repo root, and with no
    [tool.hatch.build.targets.wheel] packages declaration, hatchling can't
    infer what to ship -- pip install -e . fails at the metadata-generation
    step before it ever resolves a single dependency, regardless of whether
    the dependency list itself is correct.
    """
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    packages = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages")
    )
    assert packages, "pyproject.toml must declare [tool.hatch.build.targets.wheel] packages, or pip install -e . fails at metadata generation"
    assert "src" in packages, "src/ is where the actual code lives (imported as `from src.xxx import yyy` throughout)"

"""Code-repair crew: analyzer -> fixer -> reviewer, gated by an executed verifier.

Uses real CrewAI Agent/Task/Crew objects. Three stages run in sequence:

  1. analyzer  -- enumerates the concrete inputs that break the function
  2. fixer     -- returns corrected source
  3. reviewer  -- comments on the patch, advisory only

The load-bearing component is none of those. `verify_patch()` *executes* the
proposed patch in throwaway subprocesses (compile, import, repro, no-op, and a
non-finite-return check) and that verdict alone decides whether a patch may be
written. Small models routinely emit plausible-but-wrong patches, and during
development one confidently called a docstring-only rewrite "safe to apply, no
remaining risk" while the actual crash was untouched. The model review is
printed for context and cannot overturn the executed gates.

A patch is never written unless --apply is passed AND the verifier passes.

Run:
    .venv/Scripts/python.exe crews/code_repair_crew.py <file.py> [--repro "expr"]
"""

import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ollama serves on 127.0.0.1. CrewAI derives the provider base_url from
# OLLAMA_HOST, and "0.0.0.0" is a bind address rather than a routable client
# target, so it fails with a bare ConnectionError if left that way.
OLLAMA_BASE_URL = os.environ.get("CREW_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")

# Model routing: OpenRouter primary, local Ollama fallback.
#
# Benchmarked on a 4-bug set (empty-list division, size=0 range, swallowed
# exception, check-then-act race), scored by execution rather than by reading
# the answer:
#
#   poolside/laguna-s-2.1:free  4/4 detect, PASS as fixer, 7-21s, no GPU use
#   qwen2.5-coder:7b (local)    4/4 detect, 25s total, needs the GPU free
#   qwen3.8-27b:free            429 rate-limited (shared free pool saturated)
#   cohere/north-mini-code:free FAIL as fixer -- swapped a ZeroDivisionError for
#                              a different ValueError, so it still crashed
#   nvidia/nemotron-3-super-120b "PASS" only by returning float('nan')
#
# Two findings worth keeping: a code-specialist model and a 120B model both did
# *worse* than laguna here, so parameter count did not predict quality; and the
# free pool rate-limits hard, which is why the fallback below is not optional.
PRIMARY = os.environ.get("CREW_MODEL", "openrouter/poolside/laguna-s-2.1:free")
FALLBACK = os.environ.get("CREW_MODEL_FALLBACK", "ollama/qwen2.5-coder:7b-instruct")

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def read_openrouter_key() -> str:
    """Load OPENROUTER_API_KEY from the environment, then the Windows registry.

    The registry fallback matters in practice: the key is usually installed at
    User scope, which is invisible to an already-running process such as VS
    Code or a shell that was open beforehand. Without this, every primary model
    would fail and the crew would silently run on the weaker local model.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
                value, _ = winreg.QueryValueEx(handle, "OPENROUTER_API_KEY")
            return str(value).strip()
        except OSError:
            return ""
    return ""


def _llm(spec: str) -> LLM:
    """Build an LLM from a 'provider/model' spec, e.g. 'openrouter/x:free'.

    OpenRouter needs custom_openai=True rather than the bare "openrouter/"
    prefix. CrewAI's LLM.__new__ routes on a hardcoded provider map, but the
    "openrouter" entry has no accompanying model list, so _validate_model_in_
    constants rejects every real model ID and __new__ falls through to the
    LiteLLM branch, which is not installed:

        ImportError: ... did not match any supported native provider ...
        and the LiteLLM fallback package is not installed.

    custom_openai=True forces the native OpenAI-compatible client and skips
    that validation entirely, which is correct here because OpenRouter *is* an
    OpenAI-compatible endpoint.
    """
    if spec.startswith("openrouter/"):
        key = read_openrouter_key()
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. For this session use "
                '$env:OPENROUTER_API_KEY="sk-or-..."; to persist use '
                '[Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", '
                '"sk-or-...", "User").'
            )
        return LLM(
            model=spec,
            api_key=key,
            custom_openai=True,
            base_url="https://openrouter.ai/api/v1",
        )
    # Ollama ignores the bearer token, but the OpenAI-compatible client
    # requires one to be present.
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    return LLM(model=spec, base_url=OLLAMA_BASE_URL)


def fallback_llm(primary: str, fallback: str) -> LLM:
    """Return a real LLM whose `call` degrades to `fallback` on any failure.

    CrewAI's `Crew.kickoff()` has no per-call fallback, so a single 429 from
    the OpenRouter free tier would abort an entire run. Patching `call` on the
    instance keeps the real Agent/Task/Crew orchestration while degrading
    gracefully, and `create_llm()` returns an existing LLM instance untouched,
    so `Agent(llm=...)` accepts the result normally.

    Why a factory that patches, rather than a subclass
    --------------------------------------------------
    Two things rule out subclassing LLM here, both found the hard way:

    1. `LLM.__new__` is a factory that returns a *provider-specific* class
       (`OpenAICompletion`, `OpenAICompatibleCompletion`, ...), not the class
       it was called on. Python then discards any subclass, so a FallbackLLM
       subclass silently became a plain provider object and lost every method
       it defined.

    2. CrewAI rejects OpenRouter model ids in that same factory:

           ImportError: ... did not match any supported native provider ...
           and the LiteLLM fallback package is not installed.

       Its provider map has an "openrouter" entry but no model list beside it,
       so _validate_model_in_constants rejects the id and falls through to the
       uninstalled LiteLLM branch. `_llm()` avoids this with
       custom_openai=True plus an explicit base_url, which skips validation and
       is correct because OpenRouter *is* an OpenAI-compatible endpoint.

    Patching an already-constructed instance sidesteps both problems: the
    factory has done its work, and only `call` is replaced afterwards.

    `state` is an optional dict that records whether a fallback occurred, so
    callers can report degradation without reaching into the LLM.
    """
    state: dict[str, object] = {"fallback_used": False, "primary": primary}

    delegate = _llm(primary)
    fallback_delegate: LLM | None = None
    original_call = delegate.call

    def call_with_fallback(*args, **kwargs):
        nonlocal fallback_delegate
        try:
            return original_call(*args, **kwargs)
        except Exception as exc:  # any error here is a fallback trigger
            print(f"  [fallback] {primary} failed ({type(exc).__name__}); using {fallback}")
            state["fallback_used"] = True
            if fallback_delegate is None:
                fallback_delegate = _llm(fallback)
            return fallback_delegate.call(*args, **kwargs)

    object.__setattr__(delegate, "call", call_with_fallback)
    object.__setattr__(delegate, "fallback_state", state)
    return delegate


def extract_code(text: str) -> str:
    """Pull Python source out of a model response.

    Models wrap code in prose and fences even when told not to, so the largest
    fenced block wins when one is present.
    """
    blocks = FENCE_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


def verify_patch(original: str, patched: str, repro: str | None) -> tuple[bool, str]:
    """Execute the proposed patch and decide whether it is real.

    Gates, each in a subprocess so a syntax error or a bad import cannot take
    down this process:
      1. compiles  -- the patch is valid Python
      2. imports   -- the module actually loads
      3. repro     -- the caller's expression no longer raises (if provided),
                      and does not return NaN/inf
      4. changed   -- the patch is not a no-op

    Returns (passed, detail). Detail carries the evidence either way, so the
    verdict is auditable rather than a bare boolean.
    """
    checks: list[tuple[str, bool, str]] = []

    with tempfile.TemporaryDirectory() as tmp:
        mod_path = Path(tmp) / "candidate.py"
        mod_path.write_text(patched, encoding="utf-8")

        compile_run = subprocess.run(
            [sys.executable, "-c", f"compile(open({str(mod_path)!r}).read(), 'c', 'exec')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        checks.append((
            "compiles",
            compile_run.returncode == 0,
            (compile_run.stderr or "").strip()[-400:],
        ))

        # Import via a path-based loader so any module name in the source is
        # irrelevant to loading it.
        import_run = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util;"
                f"spec=importlib.util.spec_from_file_location('candidate',{str(mod_path)!r});"
                "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
                "print('IMPORT_OK')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        checks.append((
            "imports",
            import_run.returncode == 0 and "IMPORT_OK" in import_run.stdout,
            (import_run.stderr or import_run.stdout or "").strip()[-400:],
        ))

        if repro:
            # Two subtleties, both learned the hard way:
            #   exec(code, {}) discards the definitions, so the repro would
            #     always raise NameError and the gate would reject every patch,
            #     including correct ones.
            #   exec(code, ns) binds names into ns but leaves the -c module's own
            #     globals untouched, so a bare `average([])` still cannot see
            #     them. Evaluating the repro *inside* ns fixes both.
            repro_run = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "ns = {}\n"
                    f"exec(open({str(mod_path)!r}).read(), ns)\n"
                    f"print('REPRO_RESULT:', repr(eval({repro!r}, ns)))",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            ok = repro_run.returncode == 0
            stdout = repro_run.stdout.strip()
            detail = stdout[:200] if ok else (
                f"stdout={stdout[:200]} stderr={(repro_run.stderr or '').strip()[-300:]}"
            )
            # Reject NaN/inf on a path that previously raised. A 120B model
            # "fixed" a ZeroDivisionError by returning float('nan'), which sails
            # past a no-raise check and then propagates silently: in this repo
            # `if size > 0` is False for NaN, so the trade just disappears.
            # Crashing is louder and safer than NaN.
            if ok and ("nan" in stdout.lower() or "inf" in stdout.lower()):
                ok = False
                detail = f"returned a non-finite value (nan/inf) instead of raising: {detail}"
            checks.append(("repro", ok, detail))

    # Regression guard: a patch identical to the original would sail through
    # every gate above while fixing nothing.
    if patched.strip() == original.strip():
        checks.append(("changed_something", False, "patch is identical to the original"))

    lines = []
    for name, ok, msg in checks:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if msg and (not ok or name == "repro"):
            line += f" -- {msg}"
        lines.append(line)

    return all(ok for _, ok, _ in checks), "\n".join(lines)


def unified_diff(original: str, patched: str, rel_path: str) -> str:
    """Render the proposal as a diff the user can inspect and apply."""
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze and repair a Python file.")
    ap.add_argument("target", help="Path to the Python file to review")
    ap.add_argument(
        "--repro",
        default=None,
        help='Expression to evaluate after patching, e.g. "average([])".',
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write the patch only if the verifier passes. Off by default.",
    )
    args = ap.parse_args()

    raw = Path(args.target)
    target = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
    if not target.is_file():
        print(f"ERROR: no such file: {target}")
        return 2

    original = target.read_text(encoding="utf-8")
    rel = (
        str(target.relative_to(PROJECT_ROOT))
        if str(target).startswith(str(PROJECT_ROOT))
        else str(target)
    )
    # Diff headers are posix-style even on Windows, so the printed diff can be
    # pasted straight into `git apply`.
    diff_rel = rel.replace("\\", "/")

    analyzer_llm = fallback_llm(PRIMARY, FALLBACK)
    fixer_llm = fallback_llm(PRIMARY, FALLBACK)
    reviewer_llm = fallback_llm(PRIMARY, FALLBACK)

    analyzer = Agent(
        role="Bug analyzer",
        goal="Enumerate the concrete inputs that break this function",
        backstory="Meticulous Python reviewer. You name precise exceptions rather than speculating.",
        llm=analyzer_llm,
    )
    fixer = Agent(
        role="Code fixer",
        goal="Return corrected Python source",
        backstory="You return only corrected code inside a single python fence. No prose.",
        llm=fixer_llm,
    )
    reviewer = Agent(
        role="Fix reviewer",
        goal="Judge a proposed patch against execution evidence and name remaining risk",
        backstory="You are skeptical. Execution evidence outranks any claim of correctness.",
        llm=reviewer_llm,
    )

    # Enumeration, not judgement. The earlier wording ("identify the defect...
    # if the code is correct, say NO DEFECT FOUND") scored 0/4 on the
    # benchmark: a 7B model took that escape hatch every time, and on one
    # concurrency bug it argued the code was thread-safe when it had a
    # check-then-act race. Asking it to *list failing inputs* is a task it
    # cannot fake, and the same models then scored 4/4.
    t_analyze = Task(
        description=(
            f"Code:\n```python\n{original}\n```\n"
            "Enumerate every input that makes this function raise, and give the exact "
            "exception for each. Then list any input where it returns a silently wrong "
            "result rather than crashing, and any concurrency or mutation-ordering "
            "problem. Do not judge the code as a whole; answer only with concrete "
            "inputs, outputs and exceptions."
        ),
        expected_output="A list of concrete failing inputs with their exceptions.",
        agent=analyzer,
        max_tokens=600,
    )
    t_fix = Task(
        description=(
            f"Code:\n```python\n{original}\n```\n"
            "Return the corrected version of this code. Fix only the defects listed "
            "below; keep the same signature, docstring style and naming. Return the "
            "complete code inside one ```python fence and nothing else."
        ),
        expected_output="Corrected Python source in a single fenced block.",
        agent=fixer,
        max_tokens=800,
    )

    print("Crew: analyze -> fix (sequential)")
    Crew(
        agents=[analyzer, fixer],
        tasks=[t_analyze, t_fix],
        process=Process.sequential,
        verbose=False,
    ).kickoff()

    diagnosis = str(t_analyze.output.raw)
    # Hand the diagnosis to the fixer in a second, separate Crew. The original
    # source is deliberately absent from this context: a model shown a code
    # block right before being asked to "return the corrected code" frequently
    # echoes it back unchanged, which wastes the call and the tokens.
    t_fix.description = (
        f"An analyzer reported these defects:\n{diagnosis[:1500]}\n\n"
        "Write the corrected function. Keep the same signature, docstring style and "
        "naming. Fix only the reported defects. Return the complete corrected code "
        "inside one ```python fence and nothing else."
    )
    print("Crew: fix (analyzer findings in context)")
    Crew(
        agents=[fixer],
        tasks=[t_fix],
        process=Process.sequential,
        verbose=False,
    ).kickoff()

    patched = extract_code(str(t_fix.output.raw))

    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(diagnosis.strip()[:1200])

    if patched.strip() == original.strip():
        print("\nNo patch produced (fixer echoed the original). Nothing to verify.")
        return 1

    # The gates run Python, not the model. This verdict is what counts.
    passed, detail = verify_patch(original, patched, args.repro)

    # Only now does a model comment, and only against recorded evidence. It is
    # advisory: a model that talks itself into "looks good" cannot overturn a
    # FAIL, and one that panics about a PASS does not block it.
    t_review = Task(
        description=(
            f"Original:\n```python\n{original}\n```\n"
            f"Proposed patch:\n```python\n{patched}\n```\n"
            "Execution evidence:\n"
            f"{detail}\n\n"
            "State whether the patch is safe to apply and name any remaining risk."
        ),
        expected_output="A verdict on the patch plus any remaining risk.",
        agent=reviewer,
        max_tokens=400,
    )
    print("Crew: review (advisory)")
    Crew(
        agents=[reviewer],
        tasks=[t_review],
        process=Process.sequential,
        verbose=False,
    ).kickoff()

    print()
    print("=" * 70)
    print("VERIFIER (executed, not asked) -- this is the verdict")
    print("=" * 70)
    print(detail)
    print()
    print("=" * 70)
    print("MODEL REVIEW (advisory only)")
    print("=" * 70)
    print(str(t_review.output.raw).strip()[:900])
    print()
    print("=" * 70)
    print("PROPOSED DIFF (not written)")
    print("=" * 70)
    print(unified_diff(original, patched, diff_rel))

    if passed and args.apply:
        target.write_text(patched, encoding="utf-8")
        print(f"\nAPPLIED to {rel} (verifier passed). Review the diff before committing.")
    elif passed:
        print("\nVerifier PASSED. Re-run with --apply to write the patch.")
    else:
        print("\nVerifier FAILED -- patch NOT written. The diagnosis is the useful output here.")

    if any(x.fallback_state["fallback_used"] for x in (analyzer_llm, fixer_llm, reviewer_llm)):
        print(f"\nNote: at least one stage fell back to {FALLBACK}; the OpenRouter free tier is rate-limited.")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

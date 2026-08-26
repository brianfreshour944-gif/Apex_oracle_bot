"""
diff_utils.py

Extracts a unified diff from raw LLM output and enforces the Diff Lock
(size limits) before anything is applied to the sandbox.
"""

import re
from dataclasses import dataclass


@dataclass
class DiffResult:
    ok: bool
    diff_text: str | None
    reason: str | None
    files_touched: int = 0
    lines_changed: int = 0


def extract_diff(model_output: str) -> str | None:
    # 1. Fenced ```diff block
    match = re.search(r"```diff\n(.*?)\n```", model_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Bare unified diff fallback
    match = re.search(r"(--- a/\S+.*?(?=\n\n|\Z))", model_output, re.DOTALL)
    if match and "+++" in match.group(1):
        return match.group(1).strip()

    return None


def validate_diff(diff_text: str, max_files: int = 5, max_lines: int = 200) -> DiffResult:
    # Count files from git headers only (fall back to --- a/ markers).
    # NOTE: never count both patterns — they describe the same file.
    files_touched = len(re.findall(r"^diff --git", diff_text, re.MULTILINE))
    if files_touched == 0:
        files_touched = len(set(re.findall(r"^--- a/(\S+)", diff_text, re.MULTILINE)))

    added = len(re.findall(r"^\+(?!\+\+)", diff_text, re.MULTILINE))
    removed = len(re.findall(r"^-(?!-)", diff_text, re.MULTILINE))
    lines_changed = added + removed

    if files_touched == 0:
        return DiffResult(False, diff_text, "No file changes detected in diff.",
                          0, lines_changed)

    if files_touched > max_files:
        return DiffResult(
            False, diff_text,
            f"Patch touches {files_touched} files (limit: {max_files}). "
            "Narrow the scope to a single module.",
            files_touched, lines_changed)

    if lines_changed > max_lines:
        return DiffResult(
            False, diff_text,
            f"Patch changes {lines_changed} lines (limit: {max_lines}). "
            "Produce a smaller, more targeted fix.",
            files_touched, lines_changed)

    return DiffResult(True, diff_text, None, files_touched, lines_changed)


def get_valid_diff(model_output: str, max_files: int = 5, max_lines: int = 200) -> DiffResult:
    diff_text = extract_diff(model_output)
    if not diff_text:
        return DiffResult(
            False, None,
            "No valid diff found in model output. Ask the model to output "
            "a unified diff inside a ```diff ... ``` block.")
    return validate_diff(diff_text, max_files=max_files, max_lines=max_lines)


if __name__ == "__main__":
    sample = """
Here is the fix:

```diff
--- a/src/risk_gate.py
+++ b/src/risk_gate.py
@@ -10,7 +10,7 @@
-    if size > 0.05:
+    if size > max_position_pct:
         raise RiskLimitExceeded()
```
"""
    result = get_valid_diff(sample)
    print(result)
    assert result.ok and result.files_touched == 1 and result.lines_changed == 2
    assert get_valid_diff("no diff here").ok is False
    big = "\n".join(["+x"] * 250)
    assert not get_valid_diff(f"```diff\n{big}\n```").ok
    print("All self-tests passed.")

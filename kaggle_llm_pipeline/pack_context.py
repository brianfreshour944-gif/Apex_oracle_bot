"""
pack_context.py

Packs selected repo files + a traceback into a single prompt for the
Kaggle-hosted llama-server, with a token-budget check BEFORE sending.

Usage:
    python pack_context.py --paths "src/risk_gate.py,src/execution.py" \
        --traceback traceback.txt --question "why does X happen?" \
        > prompt.txt
"""

import argparse
import subprocess
import sys

try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    ENCODER = None


def count_tokens(text: str) -> int:
    if ENCODER:
        return len(ENCODER.encode(text))
    return len(text) // 4


def run_repomix(include_paths: str) -> str:
    result = subprocess.run(
        ["npx", "-y", "repomix", "--include", include_paths, "--stdout"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def build_prompt(system: str, repo_context: str, traceback: str, question: str) -> str:
    return (
        f"<SYSTEM>\n{system}\n</SYSTEM>\n\n"
        f"<REPO>\n{repo_context}\n</REPO>\n\n"
        f"<TRACEBACK>\n{traceback}\n</TRACEBACK>\n\n"
        f"<QUESTION>\n{question}\n</QUESTION>\n\n"
        f"Respond with brief analysis, then output the fix as a unified diff "
        f"inside a ```diff code block. Keep the patch minimal and scoped to "
        f"a single module where possible."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--traceback", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Must match llama-server -c value")
    parser.add_argument(
        "--system",
        default=(
            "You are a senior trading systems architect reviewing an "
            "algorithmic trading bot. Be precise. Never guess at code you "
            "cannot see in the provided context."
        ),
    )
    args = parser.parse_args()

    repo_context = run_repomix(args.paths)
    traceback_text = open(args.traceback).read()
    prompt = build_prompt(args.system, repo_context, traceback_text, args.question)

    total_tokens = count_tokens(prompt)
    input_budget = int(args.max_tokens * 0.7)  # leave room for output + diff

    print(prompt)
    if total_tokens > input_budget:
        print(
            f"WARNING: {total_tokens} tokens exceeds {input_budget}-token budget "
            f"(70% of {args.max_tokens}). Narrow --paths or the model may "
            f"truncate mid-file and hallucinate a fix.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[Context: {total_tokens} tokens, within budget]", file=sys.stderr)


if __name__ == "__main__":
    main()

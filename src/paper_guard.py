"""Fail-closed paper-trading guard.

The bot must never place an order against a live Alpaca endpoint. This module is
the single source of truth for that decision. It is deliberately small and
dependency-free (stdlib only) so it can be imported from settings validation,
from exchange-client construction, and from the order-submission boundary
without risking an import cycle.

Three layers, all fail-closed:

1. ``validate_configured_endpoint`` -- rejects a live/invalid configured URL
   during settings validation, before the process can start.
2. ``assert_paper_endpoint`` -- checks the endpoint the order-placement client
   ACTUALLY resolved (``TradingClient._base_url``), not just the configured
   string, so a URL that disagrees with the client's own paper flag cannot slip
   through. URL normalization and trailing slashes are handled by parsing the
   host, not by string comparison.
3. The same assertion is called immediately before every ``submit_order``, so
   even a client that was constructed some other way cannot submit.

Fail-closed means: missing, empty, malformed, or unrecognised -> NOT paper. Only
an exact match on the paper host is accepted. There is no default-to-paper and
no silent fallback: an unrecognised endpoint is refused, never assumed safe.

This guard is about the ENDPOINT. It is not the same as verifying the
ACCOUNT. ``verify_account_for_order_placement`` is a separate check used at
startup: it refuses order placement unless the authenticated account is usable
(unblocked, non-dead status) AND corroborated as a paper account. Alpaca exposes
no authoritative "is paper" boolean on the account object, so the account-number
``PA`` prefix is treated as corroborating evidence only; a missing or ambiguous
account response is NOT counted as verified (fail-closed).
"""

from __future__ import annotations

from urllib.parse import urlparse

# Exact hosts. Alpaca's paper and live trading endpoints.
PAPER_HOST = "paper-api.alpaca.markets"
LIVE_HOST = "api.alpaca.markets"

PAPER_BASE_URL = f"https://{PAPER_HOST}"


class PaperTradingViolation(RuntimeError):
    """Raised when a live/non-paper endpoint is about to be used for orders."""


def classify_endpoint(raw: object) -> str:
    """Classify an endpoint as ``"paper"``, ``"live"``, or ``"invalid"``.

    Never assumes paper. Accepts a raw string, ``None``, or an object carrying a
    ``.value`` (e.g. the alpaca-py ``BaseURL`` enum that the SDK stores on the
    client). Parses the HOST so that a trailing slash, a path, a missing scheme,
    surrounding quotes, or different casing do not change the answer. A host
    that merely CONTAINS a known host (e.g. ``paper-api.alpaca.markets.evil.com``)
    is treated as ``"invalid"``, not as paper.
    """
    if raw is None:
        return "invalid"
    value = getattr(raw, "value", None)
    s = value if isinstance(value, str) and value else str(raw)
    s = s.strip().strip("()\"' ")
    if not s:
        return "invalid"
    try:
        parsed = urlparse(s if "://" in s else "https://" + s)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return "invalid"
    if not host:
        return "invalid"
    if host == PAPER_HOST:
        return "paper"
    if host == LIVE_HOST:
        return "live"
    return "invalid"


def validate_configured_endpoint(raw: object) -> str:
    """Return the configured URL if it is the paper endpoint, else raise.

    Used from settings validation so a live or malformed ``ALPACA_BASE_URL``
    fails at startup, before any order-placement code can run.
    """
    kind = classify_endpoint(raw)
    if kind == "paper":
        return str(raw)
    raise ValueError(
        f"ALPACA_BASE_URL must be the paper endpoint {PAPER_BASE_URL!r}; got "
        f"{_safe_endpoint_label(raw)} ({kind}). Live trading is disabled in this "
        f"build. Set ALPACA_BASE_URL={PAPER_BASE_URL} (or leave it unset to use "
        f"the default)."
    )


def _safe_endpoint_label(raw: object) -> str:
    """Describe an endpoint for a log/error without disclosing credentials.

    A URL may embed credentials (``https://key:secret@host``); echoing the raw
    value would leak them. We report only the scheme, the host, and whether user
    info was present -- never the userinfo itself.
    """
    if raw is None:
        return "<none>"
    value = getattr(raw, "value", None)
    s = value if isinstance(value, str) and value else str(raw)
    s = s.strip().strip("()\"' ")
    if not s:
        return "<empty>"
    try:
        parsed = urlparse(s if "://" in s else "https://" + s)
    except ValueError:
        return "<unparseable>"
    host = parsed.hostname or "<no-host>"
    scheme = parsed.scheme or "?"
    creds = " (with embedded credentials REDACTED)" if (parsed.username or parsed.password) else ""
    return f"{scheme}://{host}{creds}"


def assert_paper_endpoint(endpoint: object, context: str = "order submission") -> None:
    """Raise unless ``endpoint`` is the paper endpoint. Never returns for live.

    ``context`` is included in the message so a refusal is traceable to the
    layer that caught it. The endpoint is reported via ``_safe_endpoint_label``
    so any embedded credentials are never written to logs.
    """
    kind = classify_endpoint(endpoint)
    if kind != "paper":
        raise PaperTradingViolation(
            f"Refusing {context}: resolved endpoint {_safe_endpoint_label(endpoint)} "
            f"is {kind}, not paper ({PAPER_HOST}). The bot is paper-only; no order "
            f"was submitted."
        )


def _mask_account_number(account_number: object) -> str:
    """Return a non-reversible label for an account number, for logs.

    Never emit the full account number: it is sensitive account information.
    We keep only the first two characters (e.g. the ``PA`` paper prefix, which is
    not secret) and the length, which is enough to identify the shape of the
    value in a log without disclosing it.
    """
    s = str(account_number).strip()
    if not s:
        return "<empty>"
    prefix = s[:2]
    return f"{prefix}*** (len={len(s)})"


def looks_like_paper_account(account_number: object) -> bool | None:
    """Best-effort paper-account check. ``None`` means "cannot tell".

    Alpaca paper account numbers begin with ``PA``. This is a heuristic for
    defence-in-depth, NOT a guarantee, and it is intentionally the weakest of
    the checks: a ``None`` (missing/unparseable) result must never be treated as
    "paper". Returns True/False only when the number is present.
    """
    if account_number is None:
        return None
    s = str(account_number).strip()
    if not s:
        return None
    return s.upper().startswith("PA")


# Account statuses that mean the account cannot place orders. Deliberately a
# blocklist of clearly-dead states rather than an allowlist of "good" ones: an
# unfamiliar-but-active status should not by itself block a legitimate paper
# account, whereas a CLOSED/DISABLED/REJECTED account must never be used.
_UNUSABLE_ACCOUNT_STATUSES = frozenset(
    {
        "ACCOUNT_CLOSED",
        "DISABLED",
        "REJECTED",
        "SUBMISSION_FAILED",
        "INACTIVE",
    }
)


def verify_account_for_order_placement(account: object) -> tuple[bool, str]:
    """Decide whether the authenticated account may place orders. Fail-closed.

    Returns ``(ok, reason)``. This is the ACCOUNT check, which is deliberately
    separate from the ENDPOINT check: the endpoint tells you which venue orders
    would reach; this tells you whether the account behind it is usable and
    consistent with a paper account.

    What is authoritative and what is not:

    * ``account_blocked`` / ``trading_blocked`` / ``trade_suspended_by_user``
      are authoritative tradability flags. Any of them set -> refuse.
    * ``status`` in a clearly-dead state (closed/disabled/rejected/...) -> refuse.
    * Alpaca exposes NO authoritative "is this a paper account" boolean on the
      account object, and the SDK does not store one on the client either. The
      account number's ``PA`` prefix is a CORROBORATING heuristic, not proof.
    * A missing or ambiguous account response therefore does NOT count as
      verified: per the fail-closed rule, "cannot tell" -> refuse. This is what
      stops the bot from declaring its environment verified when it cannot
      actually confirm it.
    """
    if account is None:
        return False, "no account response from Alpaca"

    for flag in ("account_blocked", "trading_blocked", "trade_suspended_by_user"):
        if getattr(account, flag, None):
            return False, f"{flag}=True"

    status = getattr(account, "status", None)
    status_value = getattr(status, "value", status)
    status_str = str(status_value).strip().upper() if status_value is not None else ""
    if status_str in _UNUSABLE_ACCOUNT_STATUSES:
        return False, f"account status {status_str}"

    number = getattr(account, "account_number", None)
    if number is None or not str(number).strip():
        return False, (
            "account_number missing or empty; cannot corroborate a paper account "
            "(ambiguous response is not treated as verified)"
        )
    if looks_like_paper_account(number) is not True:
        return False, (
            f"account_number {_mask_account_number(number)} is not a paper "
            "(PA...) account"
        )

    return True, "paper account corroborated (PA-prefixed, unblocked, usable status)"

#!/usr/bin/env python3
"""Check all hyperlinks in the LaTeX CV and report dead ones.

Parses every ``\\href{URL}{...}`` and ``\\url{URL}`` in the source, skips
commented-out lines and ``mailto:`` links, then makes a network request to each
unique URL. Results are classified as:

  * OK    -> 2xx / 3xx response.
  * DEAD  -> 404/410, DNS failure, connection refused, or other hard failure.
            These FAIL the build.
  * WARN  -> 401/403/405/429/999/5xx or timeout. These are typically anti-bot
            blocks (LinkedIn, ResearchGate, Google Scholar, ...) rather than a
            genuinely broken link, so by default they do NOT fail the build.

Exit code is non-zero if any DEAD link is found (or any WARN, when
``--fail-on-warn`` / ``CHECK_LINKS_FAIL_ON_WARN=1`` is set), so it can gate a CI
publish step.

Usage:
    python scripts/check_links.py main.tex
    python scripts/check_links.py main.tex --fail-on-warn
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Codes that are ambiguous (anti-bot, rate-limit, transient) rather than a
# genuinely missing page. These become WARN, never DEAD. Facebook/LinkedIn/
# ResearchGate/Cloudflare routinely return these to non-browser clients.
SOFT_FAIL_CODES = {400, 401, 403, 405, 406, 429, 503, 999}

# Only these HTTP codes are treated as a definitively broken/missing page.
HARD_DEAD_CODES = {404, 410}

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

TIMEOUT = 20
MAX_WORKERS = 12
RETRIES = 2


@dataclass
class Link:
    url: str
    line: int
    context: str  # trimmed source line, for human-friendly reporting


def _balanced_group(text: str, open_idx: int) -> tuple[str, int]:
    """Given index of a '{', return (inner_text, index_after_closing_brace)."""
    depth = 0
    i = open_idx
    while i < len(text):
        c = text[i]
        if c == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif c == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
        i += 1
    return text[open_idx + 1 :], len(text)


def _is_commented(source: str, pos: int) -> bool:
    """True if the command at `pos` sits behind an unescaped % on its line."""
    line_start = source.rfind("\n", 0, pos) + 1
    prefix = source[line_start:pos]
    # Unescaped % => rest of line is a comment.
    return bool(re.search(r"(?<!\\)%", prefix))


def extract_links(path: str) -> list[Link]:
    with open(path, encoding="utf-8") as fh:
        source = fh.read()

    links: list[Link] = []
    for m in re.finditer(r"\\(href|url)\s*\{", source):
        cmd_start = m.start()
        if _is_commented(source, cmd_start):
            continue
        url, _ = _balanced_group(source, m.end() - 1)
        url = url.strip()
        if not url or url.lower().startswith("mailto:"):
            continue
        line_no = source.count("\n", 0, cmd_start) + 1
        line_text = source.splitlines()[line_no - 1].strip()
        links.append(Link(url=url, line=line_no, context=line_text))
    return links


def _request(url: str, method: str) -> int:
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "User-Agent": BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    ctx = None
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
        return resp.status


def _classify_response(code: int) -> tuple[str, str]:
    if 200 <= code < 400:
        return "OK", f"HTTP {code}"
    if code in HARD_DEAD_CODES:
        return "DEAD", f"HTTP {code}"
    if code in SOFT_FAIL_CODES or code >= 500:
        return "WARN", f"HTTP {code} (likely bot-blocked / transient)"
    # Any other 4xx: ambiguous, don't break the build over it.
    return "WARN", f"HTTP {code}"


def _classify_error(err: Exception) -> tuple[str | None, str]:
    """Map a network exception to (status_or_None, detail).

    Returns status None when the error is transient and the caller should retry.
    """
    reason = getattr(err, "reason", err)
    text = str(reason).lower()
    if isinstance(reason, TimeoutError) or "timed out" in text:
        return None, "timed out"
    # Name resolution / connection refused => the host is unreachable/dead.
    if any(s in text for s in (
        "name or service not known",
        "nodename nor servname",
        "no address associated",
        "getaddrinfo failed",
        "connection refused",
    )):
        return "DEAD", f"unreachable: {reason}"
    # SSL and everything else: could not verify, but not provably dead.
    return None, f"{type(reason).__name__}: {reason}"


def check_url(url: str) -> tuple[str, str]:
    """Return (status, detail) where status is OK, WARN, or DEAD.

    HEAD is used only as a fast *positive* signal. Any non-OK HEAD outcome falls
    through to an authoritative GET, because many servers (notably Cloudflare)
    return bogus 404/405 to HEAD while serving 200 to GET.
    """
    last_detail = "could not verify"
    for _ in range(RETRIES + 1):
        # Fast path: a successful HEAD is trustworthy; failures are not.
        try:
            code = _request(url, "HEAD")
            if 200 <= code < 400:
                return "OK", f"HTTP {code}"
        except Exception:  # noqa: BLE001 — fall through to the authoritative GET
            pass

        # Authoritative GET.
        try:
            code = _request(url, "GET")
            return _classify_response(code)
        except urllib.error.HTTPError as e:
            status, detail = _classify_response(e.code)
            if status != "WARN" or e.code < 500:
                return status, detail
            last_detail = detail  # 5xx -> retry
        except urllib.error.URLError as e:
            status, detail = _classify_error(e)
            if status is not None:
                return status, detail
            last_detail = detail  # transient -> retry
        except Exception as e:  # noqa: BLE001
            last_detail = f"{type(e).__name__}: {e}"

    # Exhausted retries without a definitive answer.
    return "WARN", last_detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CV hyperlinks for dead URLs.")
    parser.add_argument("texfile", help="Path to the LaTeX source (e.g. main.tex)")
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        default=os.environ.get("CHECK_LINKS_FAIL_ON_WARN") == "1",
        help="Also fail the build on WARN (bot-blocked / unverifiable) links.",
    )
    args = parser.parse_args()

    links = extract_links(args.texfile)
    if not links:
        print("No hyperlinks found — nothing to check.")
        return 0

    # Map each unique URL to all (line, context) occurrences.
    occurrences: dict[str, list[Link]] = {}
    for link in links:
        occurrences.setdefault(link.url, []).append(link)

    print(f"Checking {len(occurrences)} unique URL(s) from {len(links)} link(s) in {args.texfile}...\n")

    results: dict[str, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(check_url, url): url for url in occurrences}
        for fut in as_completed(futures):
            url = futures[fut]
            results[url] = fut.result()

    dead: list[tuple[str, str]] = []
    warn: list[tuple[str, str]] = []
    ok_count = 0

    for url in sorted(occurrences):
        status, detail = results[url]
        if status == "OK":
            ok_count += 1
        elif status == "WARN":
            warn.append((url, detail))
        else:
            dead.append((url, detail))

    def report(title: str, items: list[tuple[str, str]]) -> None:
        if not items:
            return
        print(f"{title}")
        for url, detail in items:
            print(f"  • {url}")
            print(f"      reason: {detail}")
            for occ in occurrences[url]:
                print(f"      line {occ.line}: {occ.context}")
        print()

    report("⚠️  WARN — could not verify (likely anti-bot, not necessarily broken):", warn)
    report("❌  DEAD — these links are broken:", dead)

    print("─" * 60)
    print(f"OK: {ok_count}   WARN: {len(warn)}   DEAD: {len(dead)}")

    if dead or (args.fail_on_warn and warn):
        print("\nLink check FAILED.")
        return 1
    print("\nLink check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

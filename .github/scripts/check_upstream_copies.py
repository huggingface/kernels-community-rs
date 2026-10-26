#!/usr/bin/env python3
"""Fail when a file marked as an exact copy of kernels-community diverges from it.

Files copied verbatim start with a header (after any shebang line) such as

    # Copied from https://github.com/huggingface/kernels-community/blob/<sha>/<path>

or, in Markdown, the same text inside `<!-- ... -->`. This fetches `<path>` at
the pinned `<sha>` and compares it with the local file minus that header and the
blank line that may follow it. Files marked "Adapted from" are intentionally
different and are not checked.

Usage: python3 .github/scripts/check_upstream_copies.py [root]
"""

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HEADER_RE = re.compile(
    r"^(?:#|<!--) Copied from "
    r"https://github\.com/huggingface/kernels-community/blob/"
    r"(?P<sha>[0-9a-f]{40})/(?P<path>\S+?)(?: -->)?$"
)
RAW_URL = "https://raw.githubusercontent.com/huggingface/kernels-community/{sha}/{path}"
TIMEOUT_SECONDS = 30


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout
    return [root / name for name in out.decode().split("\0") if name]


def find_header(lines: list[str]) -> tuple[int, re.Match] | None:
    # The header is the first line, or the second when the first is a shebang.
    last = 1 if lines and lines[0].startswith("#!") else 0
    for index in range(min(last + 1, len(lines))):
        match = HEADER_RE.match(lines[index])
        if match:
            return index, match
    return None


def without_header(text: str, index: int) -> list[str]:
    """Candidate upstream contents: the header removed, with and without the
    blank separator line we add after it."""
    lines = text.split("\n")
    stripped = lines[:index] + lines[index + 1 :]
    candidates = ["\n".join(stripped)]
    if index < len(stripped) and stripped[index] == "":
        candidates.append("\n".join(stripped[:index] + stripped[index + 1 :]))
    return candidates


def fetch(sha: str, path: str) -> str:
    url = RAW_URL.format(sha=sha, path=path)
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode()


def check_copy(rel: str, text: str, index: int, match: re.Match, fetcher=fetch) -> str | None:
    """Returns a problem description, or None when the copy is exact."""
    sha, path = match["sha"], match["path"]
    try:
        upstream = fetcher(sha, path)
    except OSError as e:
        return f"{rel}: could not fetch upstream {path}@{sha[:8]}: {e}"
    if upstream in without_header(text, index):
        return None
    return (
        f"{rel}: diverged from kernels-community {path}@{sha[:8]}. Restore the "
        "upstream contents, or mark it 'Adapted from' with a 'Changes:' note."
    )


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    checked = 0
    problems = []
    for path in tracked_files(root):
        try:
            text = path.read_text()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        header = find_header(text.split("\n"))
        if header is None:
            continue
        checked += 1
        problem = check_copy(path.relative_to(root).as_posix(), text, *header)
        if problem:
            problems.append(problem)
    for problem in problems:
        print(f"::error::{problem}")
    print(f"Checked {checked} exact copies of kernels-community files; {len(problems)} diverged.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

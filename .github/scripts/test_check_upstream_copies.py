import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import check_upstream_copies as copies  # noqa: E402

SHA = "18153cd077b6541e9c4cb98ad9678db4e1253d1f"
URL = f"https://github.com/huggingface/kernels-community/blob/{SHA}/a/b.py"
UPSTREAM = "#!/usr/bin/env python3\nimport os\n"


def check(text, upstream=UPSTREAM):
    header = copies.find_header(text.split("\n"))
    assert header is not None
    return copies.check_copy("b.py", text, *header, fetcher=lambda sha, path: upstream)


def test_header_after_shebang_with_blank_separator_matches():
    assert check(f"#!/usr/bin/env python3\n# Copied from {URL}\n\nimport os\n") is None


def test_header_without_blank_separator_matches():
    assert check(f"# Copied from {URL}\nimport os\n", upstream="import os\n") is None


def test_markdown_header_matches():
    url = URL.replace("a/b.py", "x.md")
    header = copies.find_header([f"<!-- Copied from {url} -->"])
    assert header is not None
    assert header[1]["path"] == "x.md"


def test_local_edit_is_reported():
    problem = check(f"#!/usr/bin/env python3\n# Copied from {URL}\n\nimport sys\n")
    assert problem is not None and "diverged" in problem


def test_fetch_failure_is_reported():
    def fail(sha, path):
        raise OSError("404")

    text = f"# Copied from {URL}\nimport os\n"
    problem = copies.check_copy("b.py", text, *copies.find_header(text.split("\n")), fetcher=fail)
    assert problem is not None and "could not fetch" in problem


def test_adapted_and_unmarked_files_are_skipped():
    assert copies.find_header([f"# Adapted from {URL}", "# Changes: x"]) is None
    assert copies.find_header(["import os", f"# Copied from {URL}"]) is None

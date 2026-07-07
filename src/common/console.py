"""Windows consoles often default to a legacy codepage (e.g. cp949 on Korean
Windows) instead of UTF-8, which crashes any print() of text containing
characters outside that codepage (patient notes can contain "degree" signs,
arrows, etc). Call this once at the top of a script's entry point to avoid that.
"""

import sys


def configure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

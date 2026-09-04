"""
Loads GROQ_API_KEY from .env (via python-dotenv) and fails with a clear,
actionable message if it's missing.
"""

import os
import sys

from dotenv import load_dotenv

# .env lives at repo root; resolve that path explicitly rather than relying
# on python-dotenv's default cwd-search, which breaks if this module is
# imported from a script running in a different working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))


def require_groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit(
            "\nGROQ_API_KEY is not set.\n"
            "  1. Copy .env.example to .env in the repo root\n"
            "  2. Add your key: GROQ_API_KEY=your_actual_key_here\n"
            "  3. Get a free key at https://console.groq.com/keys\n"
        )
    return key


def get_groq_client():
    from groq import Groq
    return Groq(api_key=require_groq_api_key())

#!/usr/bin/env python
import os
from typing import List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#") -> List[str]:
    """Load requirements from a file.
    >>> _load_requirements(_PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http") or "@http" in ln:
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs
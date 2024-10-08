from __future__ import annotations

import os


def  is_valid_path(path: str) -> bool:
    if os.path.exists(path):
        return True
    else:
        return False

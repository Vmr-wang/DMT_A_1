from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dmt_mood_pipeline.task_entry import main_for_task as _main_for_task


def main_for_task(*args, **kwargs) -> None:
    _main_for_task(*args, **kwargs)

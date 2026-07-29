"""Throwaway helper for test_ledger.py's lock-timeout test
(task-ledgerlock-report.md) -- spawned as a REAL subprocess
(`python -m bandit_v1.tests._stress_lock_holder <ledger_dir> <hold_s>
<ready_marker_path>`), never imported or collected by pytest itself. Acquires
the "episodes" table lock and holds it for `hold_s` seconds, writing
`ready_marker_path` only AFTER it actually holds the lock -- so the parent
test can poll for that marker and never race the acquire (i.e. never risk
calling append_rows in the parent before this process actually holds the
lock, which would make the timeout assertion flaky/meaningless)."""
import sys
import time
from pathlib import Path

from bandit_v1 import ledger


def main():
    ledger_dir, hold_s, ready_marker = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    ledger.LEDGER_DIR = Path(ledger_dir)
    with ledger._TableLock("episodes"):
        Path(ready_marker).write_text("ready")
        time.sleep(hold_s)


if __name__ == "__main__":
    main()

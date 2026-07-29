"""Throwaway helper for test_ledger.py's cross-process append_rows stress
test (task-ledgerlock-report.md) -- spawned as a REAL subprocess
(`python -m bandit_v1.tests._stress_worker_append <ledger_dir> <proc_id> <n>`),
never imported or collected by pytest itself (no test_ functions here, no
`test_` prefix on the module). Appends `n` rows one at a time to
LEDGER_DIR/"episodes.parquet", each row tagged with this process's own
`proc` id and its own loop index `i`, so the parent test can detect loss
(missing (proc, i) pairs) or duplication (repeated pairs) directly."""
import sys
from pathlib import Path

from bandit_v1 import ledger


def main():
    ledger_dir, proc_id, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ledger.LEDGER_DIR = Path(ledger_dir)
    for i in range(n):
        ledger.append_rows("episodes", [{"proc": proc_id, "i": i}])


if __name__ == "__main__":
    main()

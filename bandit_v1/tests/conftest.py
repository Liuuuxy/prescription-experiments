"""Session tripwire: no test may mutate the real ledger config.yaml.
(A test once called run_null_phase without cfg_path, silently rewriting the
real file on every suite run -- this guard turns that class of bug into a
loud suite failure.)"""
import hashlib
from pathlib import Path
import pytest

_REAL_CFG = Path(__file__).resolve().parents[1] / "ledger" / "config.yaml"

def _digest():
    return hashlib.sha256(_REAL_CFG.read_bytes()).hexdigest() if _REAL_CFG.exists() else None

@pytest.fixture(scope="session", autouse=True)
def real_ledger_config_untouched():
    before = _digest()
    yield
    after = _digest()
    assert before == after, (
        "A test mutated the REAL bandit_v1/ledger/config.yaml "
        f"({before} -> {after}) -- some call is using a defaulted path."
    )

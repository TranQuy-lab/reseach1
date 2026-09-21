import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_raw_csv_path() -> Path:
    """The 8 real sample rows + 3 labelled synthetic edge-case rows.
    See fixtures/make_fixture.py for exactly which rows are real vs synthetic.
    """
    path = FIXTURE_DIR / "sample_raw.csv"
    assert path.exists(), "Run `python tests/fixtures/make_fixture.py` first"
    return path

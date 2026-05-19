from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "founder_linkedin_review_app.py"
SPEC = spec_from_file_location("founder_linkedin_review_app", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REVIEWED_COLUMN = MODULE.REVIEWED_COLUMN
first_unreviewed_index = MODULE.first_unreviewed_index
is_reviewed = MODULE.is_reviewed
next_unreviewed_index = MODULE.next_unreviewed_index


def test_is_reviewed_accepts_boolean_like_true_values() -> None:
    assert is_reviewed({REVIEWED_COLUMN: True}) is True
    assert is_reviewed({REVIEWED_COLUMN: "true"}) is True
    assert is_reviewed({REVIEWED_COLUMN: " TRUE "}) is True
    assert is_reviewed({REVIEWED_COLUMN: ""}) is False


def test_first_unreviewed_index_returns_first_non_true_row() -> None:
    rows = [
        {REVIEWED_COLUMN: "true"},
        {REVIEWED_COLUMN: True},
        {REVIEWED_COLUMN: ""},
        {REVIEWED_COLUMN: "false"},
    ]

    assert first_unreviewed_index(rows) == 2


def test_next_unreviewed_index_skips_reviewed_rows() -> None:
    rows = [
        {REVIEWED_COLUMN: "true"},
        {REVIEWED_COLUMN: "true"},
        {REVIEWED_COLUMN: ""},
        {REVIEWED_COLUMN: "false"},
    ]

    assert next_unreviewed_index(rows, 1) == 2
    assert next_unreviewed_index(rows, 3) == 3


def test_next_unreviewed_index_returns_requested_position_if_no_later_unreviewed_row() -> None:
    rows = [
        {REVIEWED_COLUMN: ""},
        {REVIEWED_COLUMN: "true"},
        {REVIEWED_COLUMN: "true"},
    ]

    assert next_unreviewed_index(rows, 1) == 1

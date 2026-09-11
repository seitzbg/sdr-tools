"""Stream record validation (R15): a bad line must be skippable, never fatal."""
import json

from sdr_sweep.record import parse_sweep, validate_sweep

GOOD = {"t": 1.0, "f0": 915e6, "f1": 915.1e6, "bin": 1e4, "clip": 0.0,
        "db": [-90.0, -80.0, -70.0]}


def test_valid_record_normalised():
    rec = parse_sweep(json.dumps(GOOD))
    assert rec is not None
    assert rec["db"] == [-90.0, -80.0, -70.0]
    assert rec["f0"] == 915e6 and rec["bin"] == 1e4
    assert rec["t"] == 1.0 and rec["clip"] == 0.0


def test_optional_fields_default():
    rec = parse_sweep(json.dumps({"f0": 1.0, "f1": 2.0, "bin": 1.0, "db": [-1.0]}))
    assert rec["t"] == 0.0 and rec["clip"] == 0.0


def test_blank_and_syntax_error_skip():
    assert parse_sweep("") is None
    assert parse_sweep("   ") is None
    assert parse_sweep("{not json") is None


def test_wrong_json_shapes_skip():
    # Valid JSON, wrong shape — the R15 cases that used to crash the consumer.
    for line in ("null", "[1, 2, 3]", "42", '"a string"', "true"):
        assert parse_sweep(line) is None


def test_bad_db_shapes_skip():
    assert validate_sweep({"f0": 1, "f1": 2, "bin": 1, "db": []}) is None       # empty
    assert validate_sweep({"f0": 1, "f1": 2, "bin": 1, "db": None}) is None      # null db
    assert validate_sweep({"f0": 1, "f1": 2, "bin": 1, "db": "abc"}) is None     # string db
    assert validate_sweep({"f0": 1, "f1": 2, "bin": 1, "db": [None]}) is None    # null value
    assert validate_sweep({"f0": 1, "f1": 2, "bin": 1, "db": ["x"]}) is None     # string value


def test_nonfinite_values_skip():
    assert validate_sweep({"f0": 1, "f1": 2, "bin": 1, "db": [float("nan")]}) is None
    assert validate_sweep({"f0": 1, "f1": 2, "bin": 1, "db": [float("inf")]}) is None
    assert validate_sweep({"f0": float("nan"), "f1": 2, "bin": 1, "db": [-1.0]}) is None


def test_missing_required_field_skips():
    assert validate_sweep({"f1": 2, "bin": 1, "db": [-1.0]}) is None    # no f0


def test_bool_not_treated_as_number():
    # True is an int subclass; it must not sneak in as 1.0.
    assert validate_sweep({"f0": True, "f1": 2, "bin": 1, "db": [-1.0]}) is None
    assert validate_sweep({"f0": 1, "f1": 2, "bin": 1, "db": [True]}) is None

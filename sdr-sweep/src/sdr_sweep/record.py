"""Shared, stdlib-only parser/validator for the sdr-sweep JSONL stream.

Both the exporter and the TUI consume one JSON object per sweep on stdin. A
single malformed line — a decode error, a `null`, a bare list, a record whose
`db` is missing/empty/non-numeric, or a non-finite value — must never terminate
the consumer: a bad line in a replay or a composed stream should be dropped so
the following valid records are still processed.

`parse_sweep(line)` returns a validated, normalised dict (all of `db`, `f0`,
`f1`, `bin` present as finite floats, `t`/`clip` defaulted) or `None` to skip.
Kept dependency-free (json + math only) so the exporter stays stdlib-only.
"""
import json
import math

_REQUIRED_NUM = ("f0", "f1", "bin")


def _finite_number(v):
    """True for a real, finite int/float — excluding bool (a subclass of int)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_sweep(line):
    """Parse one JSONL sweep record. Return a normalised dict, or None to skip.

    A valid record is a JSON *object* with finite numeric ``f0``, ``f1`` and
    ``bin`` and a non-empty ``db`` list of finite numbers. ``t`` and ``clip``
    are optional and default to 0.0. Anything else returns None so the caller
    can drop the line and continue the stream.
    """
    if not isinstance(line, str):
        return None
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    return validate_sweep(obj)


def validate_sweep(obj):
    """Validate an already-decoded record (see parse_sweep). Return dict or None."""
    if not isinstance(obj, dict):
        return None
    db = obj.get("db")
    if not isinstance(db, list) or not db:
        return None
    dbf = []
    for v in db:
        if not _finite_number(v):
            return None
        dbf.append(float(v))
    out = {"db": dbf}
    for key in _REQUIRED_NUM:
        v = obj.get(key)
        if not _finite_number(v):
            return None
        out[key] = float(v)
    t = obj.get("t", 0.0)
    out["t"] = float(t) if _finite_number(t) else 0.0
    clip = obj.get("clip", 0.0)
    out["clip"] = float(clip) if _finite_number(clip) else 0.0
    return out

"""G3: the comparator threshold registry is checkable, and nothing more.

This file is READ-ONLY with respect to the comparator and the registry: it opens
the registered JSON, re-derives every claim from the files on disk, and reports
VALID or BLOCKED_DEPENDENCY. It never edits the registry, never writes a
comparator, and never claims a numerical result - the registry itself says no
case has run, and this file holds it to that by refusing any case whose status
is anything but NOT_RUN.

WHY THE FOLD. The deployed table states thresholds as Python expressions
(2 ** -10 and friends). ast.literal_eval raises ValueError on those - measured,
even for 2 ** -10 - so a validator built on it would either crash or get "fixed"
by hard-coding numbers copied from the table. The folder below evaluates
Pow/UnaryOp/Constant exactly, and a test pins the fact that literal_eval really
does fail on the same source, so nobody can quietly put it back.

WHY THE CONSTANT BRANCH. float8_e5m2 last two fields are the literal 1.0, not an
exponent; a folder that assumed "every value is 2**n" would silently misread
them. Both shapes are evaluated by shape, and the expected quintuple is
asserted field by field.

WHY POSITION MATTERS. A case binds its threshold entry by POSITION, so the
order list is part of the meaning: a reordered entry names different fields
with the same numbers.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "docs" / "handoff" / "COMPARATOR_THRESHOLD_REGISTRY.json"

SCRIPTS = Path("/home/s00653124/workspace/BiShengKernelAgent/skills/common/"
               "ascend-npu-run-eval/scripts")
COMPARATOR = SCRIPTS / "npu_compare.py"
THRESHOLD_TABLE = SCRIPTS / "cann_bench_compare.py"

FIELDS = ("threshold", "small_value_threshold", "small_value_error",
          "cancel_boundary", "cancel_zero_threshold")

EXPECTED = {
    "float16": (2 ** -10, 2 ** -11, 2 ** -16, 2 ** -5, 2 ** -5),
    "bfloat16": (2 ** -7, 2 ** -8, 2 ** -16, 2 ** -3, 2 ** -3),
    "float32": (2 ** -13, 2 ** -14, 2 ** -30, 2 ** -8, 2 ** -8),
    "float64": (2 ** -13, 2 ** -14, 2 ** -30, 2 ** -8, 2 ** -8),
    "hifloat32": (2 ** -11, 2 ** -12, 2 ** -28, 2 ** -8, 2 ** -8),
    "float8_e4m3": (2 ** -3, 2 ** -4, 2 ** -6, 2 ** -1, 2 ** -1),
    # the two constant entries: 2 ** 0 would be a different, wrong reading
    "float8_e5m2": (2 ** -2, 2 ** -3, 2 ** -5, 1.0, 1.0),
}

BLOCKED = "BLOCKED_DEPENDENCY"
#: The artifact version this file validates. A different version is a different
#: contract, and guessing at its fields is how a validator certifies a shape it
#: was never written for.
EXPECTED_SCHEMA = "tf-ascend.comparator-registry.v2"


class Blocked(RuntimeError):
    """A named refusal: a dependency the registry rests on is not what it says."""

    def __init__(self, code: str, detail: str, where: str = "") -> None:
        self.code = code
        self.detail = detail
        self.where = where
        super().__init__("%s: %s%s" % (code, detail,
                                      "" if not where else " (%s)" % where))


def fold(expression: str):
    """Evaluate the tiny expression language the threshold table uses.

    Only the shapes the deployed table actually contains are accepted, and every
    other shape is refused BY NAME with the source text: this is not a general
    Python evaluator, and pretending it were is how a threshold gets invented.
    """
    try:
        tree = ast.parse(str(expression).strip(), mode="eval")
    except SyntaxError as error:
        raise Blocked(BLOCKED, "threshold source %r does not parse: %s"
                      % (expression, error)) from error
    return _fold_node(tree.body, expression)


def _fold_node(node, source: str):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise Blocked(BLOCKED, "threshold source %r holds a non-numeric constant"
                          % (source,))
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_fold_node(node.operand, source)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return +_fold_node(node.operand, source)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        base = _fold_node(node.left, source)
        exponent = _fold_node(node.right, source)
        if not isinstance(base, int) or isinstance(exponent, float):
            raise Blocked(BLOCKED, "threshold source %r is not an integer base with "
                                  "an integer exponent" % (source,))
        return float(base ** exponent)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _fold_node(node.left, source) / _fold_node(node.right, source)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _fold_node(node.left, source) * _fold_node(node.right, source)
    raise Blocked(BLOCKED, "threshold source %r uses %s, which this validator "
                           "does not fold" % (source, type(node).__name__))


def _normalised(expression: str) -> str:
    """A whitespace- and parenthesis-insensitive form of one expression."""
    return ast.unparse(ast.parse(str(expression).strip(), mode="eval"))


def source_forms_of(text: str, first: int, last: int) -> dict:
    """Read the deployed table five source forms per dtype, by line range."""
    lines = text.splitlines()
    if not 1 <= first <= last <= len(lines):
        raise Blocked(BLOCKED, "declared line range [%d,%d] is outside the table "
                               "file" % (first, last), str(THRESHOLD_TABLE))
    chunk = "\n".join(lines[first - 1:last])
    try:
        tree = ast.parse(chunk, mode="exec")
    except SyntaxError as error:
        raise Blocked(BLOCKED, "the declared line range does not parse as a table: "
                               "%s" % (error,)) from error
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name)
                    and target.id == "CANN_BENCH_THRESHOLDS"):
                continue
            if not isinstance(node.value, ast.Dict):
                raise Blocked(BLOCKED, "CANN_BENCH_THRESHOLDS is not a dict literal")
            for key, value in zip(node.value.keys, node.value.values):
                if not isinstance(key, ast.Constant):
                    raise Blocked(BLOCKED, "a dtype key is not a literal")
                if not isinstance(value, ast.Dict):
                    raise Blocked(BLOCKED, "dtype %r is not a dict literal"
                                  % (key.value,))
                fields = {}
                for field_key, field_value in zip(value.keys, value.values):
                    if not isinstance(field_key, ast.Constant):
                        raise Blocked(BLOCKED, "a field name is not a literal")
                    fields[str(field_key.value)] = ast.unparse(field_value)
                found[str(key.value)] = fields
    if not found:
        raise Blocked(BLOCKED, "the declared line range holds no "
                               "CANN_BENCH_THRESHOLDS")
    return found


@dataclass(frozen=True)
class Verdict:
    status: str
    detail: str
    checked: tuple = ()


#: Every top-level field the artifact must carry, with why it is not optional.
REQUIRED_KEYS = {
    "schema_version": "a consumer must be able to tell which format it reads",
    "draft": "the artifact must say it is a draft, never imply closure",
    "registered_at_utc": "a registration without a time cannot be audited",
    "operator": "the operators the cases belong to",
    "purpose": "what the artifact is for, and what it is not",
    "claims_not_made": "the explicit negatives",
    "unknowns": "the honest list of what is NOT established",
    "comparator": "which comparator, by path and digest",
    "threshold_table": "which table, by path, digest and line range",
    "cases": "the cases themselves",
    "dtype_distribution": "the measured distribution the cases must match",
    "tree_anchor": "the in-repository evidence",
    "non_normalisation_note": "why the 1.0 entries are recorded as written",
    "registry_history": "the superseded artifacts and their digests",
    "binding_rules": "the stated contract for case_id/source_key/source_case_id",
}

#: Phrases an honesty section must still carry, so it cannot be hollowed out.
REQUIRED_NEGATIONS = ("no case", "not certify", "no threshold", "no allclose")


def _require_required_keys(registry: dict) -> None:
    """Every declared top-level field must be present, and none may be empty."""
    missing = sorted(key for key in REQUIRED_KEYS if key not in registry)
    if missing:
        raise Blocked(BLOCKED, "the registry is missing required top-level "
                               "field(s) %s" % (missing,), "top level")
    empty = sorted(key for key in REQUIRED_KEYS
                   if registry[key] in (None, "", [], {}))
    if empty:
        raise Blocked(BLOCKED, "the registry states required field(s) %s as empty"
                      % (empty,), "top level")


def _require_honesty_sections(registry: dict) -> None:
    """The negatives must be present AND still say what they said.

    An empty list, or a list reworded into vague noise, would satisfy "the key
    exists" while dropping the only statement of what this artifact does not
    prove. Each required negation is checked as a substring of the joined text,
    and unknowns must name at least one substantive thing that is not
    established.
    """
    claims = registry["claims_not_made"]
    if not isinstance(claims, list) or not claims:
        raise Blocked(BLOCKED, "claims_not_made is not a non-empty list", "honesty")
    joined = " ".join(str(item) for item in claims).lower()
    absent = [phrase for phrase in REQUIRED_NEGATIONS if phrase not in joined]
    if absent:
        raise Blocked(BLOCKED, "claims_not_made no longer states %s" % (absent,),
                      "honesty")
    unknowns = registry["unknowns"]
    if not isinstance(unknowns, list) or not unknowns:
        raise Blocked(BLOCKED, "unknowns is not a non-empty list: the registry "
                               "must keep stating what it does not establish",
                      "honesty")
    if not any(len(str(item)) > 40 for item in unknowns):
        raise Blocked(BLOCKED, "unknowns holds no substantive item", "honesty")


def _require_tree_anchor(registry: dict) -> None:
    """The in-repository anchor must be present and must match the tree.

    This is the only evidence INSIDE this repository that the registered
    comparator is the one the harness imports; losing it while staying VALID
    would leave the artifact citing nothing local.
    """
    anchor = registry["tree_anchor"]
    if not isinstance(anchor, dict):
        raise Blocked(BLOCKED, "tree_anchor is not a mapping", "anchor")
    for key in ("path", "sha256", "lines", "contract_lines"):
        if key not in anchor:
            raise Blocked(BLOCKED, "tree_anchor states no %s" % (key,), "anchor")
    path = Path(anchor["path"])
    if not path.is_absolute():
        path = REPO / path
    if not path.is_file():
        raise Blocked(BLOCKED, "the anchored contract file is missing", str(path))
    actual = _digest(path)
    if actual != anchor["sha256"]:
        raise Blocked(BLOCKED, "the anchored contract file digest is not the "
                               "registered one",
                      "%s != %s" % (actual, anchor["sha256"]))
    lines = anchor["lines"]
    if not isinstance(lines, list) or not lines:
        raise Blocked(BLOCKED, "tree_anchor states no line list", "anchor")
    text = path.read_text().splitlines()
    for number in lines:
        if not isinstance(number, int) or not 1 <= number <= len(text):
            raise Blocked(BLOCKED, "tree_anchor line %r is outside the file"
                          % (number,), str(path))
    contents = anchor.get("line_contents") or {}
    for key, expected in contents.items():
        number = int(key)
        if str(expected).strip() not in text[number - 1]:
            raise Blocked(BLOCKED, "line %d no longer reads %r" % (number, expected),
                          str(path))

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(directory: Path):
    """The repository head above the comparator, or None when it is not a repo.

    A missing repository is not a licence to skip the claim: the registry states
    a head, and a head that cannot be measured is reported as such by the
    caller rather than silently accepted.
    """
    import subprocess

    try:
        completed = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=False,
                                   timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def validate(registry: dict, *, comparator: Path = COMPARATOR,
             table: Path = THRESHOLD_TABLE) -> Verdict:
    """Re-derive every claim the registry makes; VALID or BLOCKED_DEPENDENCY."""
    checked = []
    # The SHAPE of the artifact is checked before any value: a registry that no
    # longer carries its honesty section or its in-repository anchor would pass a
    # value-only validation while having lost the two things it exists to record.
    # Shape runs FIRST on purpose - finding a missing field after the numeric
    # folds would mean the folds were doing the shape's job.
    _require_required_keys(registry)
    _require_honesty_sections(registry)
    _require_tree_anchor(registry)
    for path, claim, label in ((comparator, registry["comparator"]["sha256"],
                                "comparator"),
                               (table, registry["threshold_table"]["sha256"],
                                "threshold table")):
        if not path.is_file():
            raise Blocked(BLOCKED, "the %s named by the registry is missing" % label,
                          str(path))
        actual = _digest(path)
        if actual != claim:
            raise Blocked(BLOCKED, "the %s sha256 is not the registered one" % label,
                          "%s != %s" % (actual, claim))
        checked.append((label, actual))
    head = registry["comparator"].get("repo_head")
    if not head:
        raise Blocked(BLOCKED, "the registry states no comparator repo head")
    actual_head = _git_head(comparator.parent)
    if actual_head is not None and actual_head != head:
        raise Blocked(BLOCKED, "the comparator repository head is not the "
                               "registered one", "%s != %s" % (actual_head, head))
    checked.append(("repo_head", actual_head or "not-a-repository"))
    table_ref = registry["threshold_table"]
    if Path(table_ref["path"]) != table:
        raise Blocked(BLOCKED, "threshold_table.path is not the registered table",
                      table_ref["path"])
    if list(table_ref["field_order"]) != list(FIELDS):
        raise Blocked(BLOCKED, "field_order is not the deployed field order",
                      str(table_ref["field_order"]))
    forms = source_forms_of(table.read_text(), *table_ref["line_range"])
    declared = table_ref["dtype_entries"]
    if set(forms) != set(declared):
        raise Blocked(BLOCKED, "the table dtype set and the registry differ",
                      "%s vs %s" % (sorted(forms), sorted(declared)))
    for dtype, entry in declared.items():
        fields = forms[dtype]
        if set(fields) != set(FIELDS):
            raise Blocked(BLOCKED, "dtype %r does not state all five fields" % (dtype,),
                          str(sorted(fields)))
        if len(entry["source_forms"]) != len(FIELDS):
            raise Blocked(BLOCKED, "dtype %r registers %d source form(s) for %d "
                                   "fields" % (dtype, len(entry["source_forms"]),
                                               len(FIELDS)))
        for field, source in zip(FIELDS, entry["source_forms"]):
            folded = fold(source)
            if folded != fold(fields[field]):
                raise Blocked(
                    BLOCKED,
                    "dtype %r field %r is registered as %r but the table states %r"
                    % (dtype, field, folded, fold(fields[field])), source)
            # The SOURCE SHAPE is part of the claim too: 1.0 and 2 ** 0 fold to
            # the same float, so comparing values alone would accept a rewritten
            # table. The registered form must be the table's own form.
            if _normalised(source) != _normalised(fields[field]):
                raise Blocked(
                    BLOCKED,
                    "dtype %r field %r is registered with source %r but the table "
                    "writes %r" % (dtype, field, source, fields[field]), source)
        values = tuple(float(value) for value in entry["values"])
        if len(values) != len(FIELDS):
            raise Blocked(BLOCKED, "dtype %r states %d value(s) for %d fields"
                          % (dtype, len(values), len(FIELDS)))
        if values != EXPECTED[dtype]:
            raise Blocked(BLOCKED, "dtype %r registered quintuple is not the "
                                   "deployed one" % (dtype,), str(values))
        checked.append((dtype, values))
    fallback = table_ref.get("cann_bench_fallback")
    if not isinstance(fallback, dict) or "line" not in fallback:
        raise Blocked(BLOCKED, "the registry states no fallback entry")
    if int(fallback["line"]) <= int(table_ref["line_range"][1]):
        raise Blocked(BLOCKED, "the fallback line must be stated separately from "
                               "the table range", str(fallback["line"]))
    for case in registry["cases"]:
        dtype = case.get("dtype")
        if dtype not in EXPECTED:
            raise Blocked(BLOCKED, "case %r names dtype %r, which has no threshold "
                                   "entry" % (case.get("case_id"), dtype))
        entry = case.get("threshold_entry")
        if not isinstance(entry, dict) or set(entry) != {"order", "source_forms",
                                                        "values"}:
            raise Blocked(BLOCKED, "case %r threshold_entry is not the registered "
                                   "triple" % (case.get("case_id"),))
        # POSITION IS MEANING: a reordered entry names other fields with the
        # same numbers, so the order list must be the deployed one.
        if list(entry["order"]) != list(FIELDS):
            raise Blocked(BLOCKED, "case %r states a different field order"
                          % (case.get("case_id"),), str(entry["order"]))
        values = tuple(float(value) for value in entry["values"])
        if values != EXPECTED[dtype]:
            raise Blocked(BLOCKED, "case %r threshold_entry is not its dtype entry"
                          % (case.get("case_id"),))
        for field, source in zip(FIELDS, entry["source_forms"]):
            if fold(source) != EXPECTED[dtype][FIELDS.index(field)]:
                raise Blocked(BLOCKED, "case %r field %r source form disagrees with "
                                       "the deployed value"
                              % (case.get("case_id"), field), source)
        if case.get("registration_basis") in (None, ""):
            raise Blocked(BLOCKED, "case %r states no registration basis"
                          % (case.get("case_id"),))
    _require_case_set(registry)
    _require_distribution(registry)
    _require_no_pass_claim(registry)
    checked.append(("cases", len(registry["cases"])))
    return Verdict("VALID", "the registry reproduces the deployed comparator and "
                            "threshold table", tuple(checked))


#: The external indices the case set must be bound to. The registry cannot
#: certify its own completeness: authoritative_total is a claim inside the same
#: file that would have to be trusted, so the ids are compared with the case
#: matrix and the supplemental list on disk.
CASE_MATRIX = REPO / "work_package" / "cases" / "case_matrix.json"
SUPPLEMENTAL = REPO / "work_package" / "cases" / "supplemental_mla.json"


def _authoritative_ids() -> list:
    """case_matrix ids as the registry spells them: case/case_id."""
    data = json.loads(CASE_MATRIX.read_text())
    rows = data["cases"] if isinstance(data, dict) else data
    return ["%s/%s" % (row["case"], row["case_id"]) for row in rows]


#: The source files the registry may name, as PATHS THE VALIDATOR READS. A
#: source_file the registry states but this map does not know is refused: an
#: annotation the validator ignores would be a decorative field.
SOURCE_FILES = {
    "work_package/cases/case_matrix.json": CASE_MATRIX,
    "work_package/cases/supplemental_mla.json": SUPPLEMENTAL,
}


def _source_indices() -> dict:
    """Every case the source files declare, grouped BY THE FILE THEY NAME.

    The registry states source_file per case, and this function reads exactly
    that file: the annotation is bound rather than trusted, so naming a file the
    case did not come from is a refusal, not a cosmetic difference. A validator
    that read fixed paths instead would leave the field decorative.

    The supplemental source declares NO case/operator field (measured: its cases
    carry case_id/dtypes/key/purpose/status and no case), so operator is sourced
    only for the authoritative matrix; the supplemental cases are bound on key,
    case_id and dtypes, which is what that file actually states.
    """
    by_key = {}
    for name, path in SOURCE_FILES.items():
        if not path.is_file():
            raise Blocked(BLOCKED, "the source file named by the registry is "
                                   "missing" % (), str(path))
        data = json.loads(path.read_text())
        rows = data["cases"] if isinstance(data, dict) else data
        for row in rows:
            by_key[str(row["key"])] = {
                "source_file": name,
                "operator": str(row["case"]) if row.get("case") is not None else None,
                "case_id": row.get("case_id"),
                "dtypes": tuple(str(item) for item in row.get("dtypes") or ()),
            }
    return by_key

def _bind_case_ids(ids, cases) -> None:
    """Bind every case to the source index, field by field, verbatim.

    Four questions, four assertions:
      * the KEY SET is the unique global anchor (44 distinct keys);
      * REVERSE: a source_key no source states does not exist, so a fabricated
        key is refused rather than ignored;
      * per case, (operator, source_case_id) is the source (case, case_id) and the
        dtype is one the source case carries - compared PER CASE, never as a
        global set, because the authoritative ids repeat across operators;
      * case_id stays registry-local (its own contract) but must be present and
        unique, and each case supplemental marker must agree with its operator.
    """
    source = _source_indices()
    expected_keys = set(source)
    by_key = {str(case.get("source_key", "")) for case in cases}
    if by_key != expected_keys:
        raise Blocked(BLOCKED, "the registered source keys are not the source "
                               "files keys: missing=%s extra=%s"
                      % (sorted(expected_keys - by_key),
                         sorted(by_key - expected_keys)), "case set")
    if len(by_key) != len(cases):
        raise Blocked(BLOCKED, "two cases share one source key", "case set")
    unsupported = [case.get("case_id") for case in cases
                   if str(case.get("source_key", "")) not in expected_keys]
    if unsupported:
        raise Blocked(BLOCKED, "case(s) %s carry a source_key no source file "
                               "states" % (unsupported,), "case set")
    for case in cases:
        record = source.get(str(case.get("source_key", "")))
        if record is None:
            continue
        stated_file = str(case.get("source_file", ""))
        if stated_file not in SOURCE_FILES:
            raise Blocked(BLOCKED, "case %r names source_file %r, which this "
                                   "validator does not read"
                          % (case.get("case_id"), stated_file), "case set")
        if stated_file != record["source_file"]:
            raise Blocked(BLOCKED, "case %r states source_file %r but its "
                                   "source_key is declared by %r"
                          % (case.get("case_id"), stated_file,
                             record["source_file"]), "case set")
        if record["operator"] is not None and \
                str(case.get("operator")) != record["operator"]:
            raise Blocked(BLOCKED, "case %r states operator %r while the source "
                                   "case is %r"
                          % (case.get("case_id"), case.get("operator"),
                             record["operator"]), "case set")
        if str(case.get("source_case_id")) != str(record["case_id"]):
            raise Blocked(BLOCKED, "case %r states source_case_id %r while the "
                                   "source case is %r"
                          % (case.get("case_id"), case.get("source_case_id"),
                             record["case_id"]), "case set")
        if record["dtypes"] and str(case.get("dtype")) not in record["dtypes"]:
            raise Blocked(BLOCKED, "case %r states dtype %r, which the source case "
                                   "does not carry"
                          % (case.get("case_id"), case.get("dtype")), "case set")
    local = [str(case.get("case_id", "")) for case in cases]
    if any(not value for value in local) or len(set(local)) != len(local):
        raise Blocked(BLOCKED, "a case_id is empty or repeated", "case set")
    for case in cases:
        has_marker = "supplemental" in case
        is_supplemental = str(case.get("source_file", "")).endswith(
            "supplemental_mla.json")
        if has_marker != is_supplemental:
            raise Blocked(BLOCKED, "case %r states supplemental=%r, which does not "
                                   "match its operator"
                          % (case.get("case_id"), case.get("supplemental")),
                          "case set")

def _require_case_set(registry: dict) -> None:
    """The case SET is what the registry claims it enumerated.

    Deleting one case was previously caught only as a side effect of the
    distribution arithmetic, which made the failure message point at a dtype
    count instead of at the missing case. This states the real invariant: every
    case has a unique id, and the authoritative and supplementary counts are the
    declared ones, so a deletion is reported as a deletion.
    """
    cases = registry["cases"]
    if not isinstance(cases, list) or not cases:
        raise Blocked(BLOCKED, "the registry states no cases", "case set")
    ids = [str(case.get("case_id", "")) for case in cases]
    if any(not case_id for case_id in ids):
        raise Blocked(BLOCKED, "a case states no case_id", "case set")
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise Blocked(BLOCKED, "the registry repeats case id(s) %s" % (duplicates,),
                      "case set")
    declared = registry.get("dtype_distribution") or {}
    total = declared.get("authoritative_total")
    # The id SET is compared with the external indices FIRST: a deletion must be
    # reported as "this id is missing", not as an arithmetic side effect of a
    # count the same file supplies (and so cannot certify itself).
    _bind_case_ids(ids, cases)
    authoritative = 0
    supplementary_count = 0
    for case in cases:
        if str(case.get("supplemental", "") or "").strip() or \
                "supplemental" in str(case.get("operator", "")) or \
                "supplemental" in str(case.get("case_id", "")):
            supplementary_count += 1
        else:
            authoritative += 1
    if total is not None and authoritative != int(total):
        raise Blocked(BLOCKED, "the registry holds %d authoritative case(s), not "
                               "the declared %d: a case is missing or extra"
                      % (authoritative, int(total)), "case set")
    expected_supplementary = sum(
        int(count) for operator, per_dtype in declared.items()
        if operator.startswith("supplemental") and isinstance(per_dtype, dict)
        for count in per_dtype.values())
    if expected_supplementary and supplementary_count != expected_supplementary:
        raise Blocked(BLOCKED, "the registry holds %d supplementary case(s), not "
                               "the declared %d"
                      % (supplementary_count, expected_supplementary), "case set")


def _require_distribution(registry: dict) -> None:
    """The cases must be the distribution the registry declares.

    A case whose dtype has an entry in the table is still wrong if the case
    matrix never had that dtype: the entry a case uses comes from the case's own
    dtype, not from what the table happens to contain.
    """
    declared = registry.get("dtype_distribution")
    if not isinstance(declared, dict):
        raise Blocked(BLOCKED, "the registry declares no dtype distribution")
    # A supplementary case carries its own marker; it is counted under
    # supplemental_mla, which is why grouping by operator alone disagreed with
    # the declared distribution.
    counts = {}
    for case in registry["cases"]:
        operator = str(case.get("operator"))
        if str(case.get("supplemental", "") or "").strip() or \
                "supplemental" in str(case.get("case_id", "")):
            operator = "supplemental_mla"
        key = (operator, str(case.get("dtype")))
        counts[key] = counts.get(key, 0) + 1
    for operator, per_dtype in declared.items():
        if operator in ("authoritative_total", "execution_status"):
            continue
        if not isinstance(per_dtype, dict):
            raise Blocked(BLOCKED, "distribution entry %r is not a mapping"
                          % (operator,))
        for dtype, expected in per_dtype.items():
            actual = counts.get((operator, dtype), 0)
            if actual != int(expected):
                raise Blocked(
                    BLOCKED,
                    "the distribution declares %d %s/%s case(s) but the registry "
                    "holds %d" % (int(expected), operator, dtype, actual),
                    "dtype distribution")
    total = declared.get("authoritative_total")
    authoritative = sum(count for (operator, _dtype), count in counts.items()
                        if operator != "supplemental_mla")
    if total is not None and authoritative != int(total):
        raise Blocked(BLOCKED, "the registry holds %d authoritative case(s) for a "
                               "declared total of %d"
                      % (authoritative, int(total)))


def _require_no_pass_claim(registry: dict) -> None:
    """The registry may not claim a result; this validator holds it to that."""
    for case in registry["cases"]:
        status = case.get("execution_status")
        if status is not None and status != "NOT_RUN":
            raise Blocked(BLOCKED, "case %r claims execution_status %r; the registry "
                                   "may not assert a result"
                          % (case.get("case_id"), status))
    if str(registry.get("schema_version", "")) != EXPECTED_SCHEMA:
        raise Blocked(BLOCKED, "the registry states schema %r, not %r"
                      % (registry.get("schema_version"), EXPECTED_SCHEMA))
    if registry.get("draft") is not True:
        raise Blocked(BLOCKED, "the registry is not marked DRAFT")
    claims = registry.get("claims_not_made")
    if not claims or not any("no case" in str(item).lower() for item in claims):
        raise Blocked(BLOCKED, "the registry does not state that no case has run")


def _load() -> dict:
    return json.loads(REGISTRY.read_text())


class _Delete:
    pass


_DELETE = _Delete()


def _variant(**changes) -> dict:
    """A registry copy edited IN MEMORY; the file on disk is never touched."""
    registry = json.loads(REGISTRY.read_text())
    for path, value in changes.items():
        node = registry
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[int(part)] if part.isdigit() else node[part]
        key = parts[-1]
        if value is _DELETE:
            node.pop(int(key) if key.isdigit() else key)
        else:
            node[int(key) if key.isdigit() else key] = value
    return registry


def test_the_registry_reproduces_the_deployed_table():
    """POSITIVE: the draft validates against the real files and claims nothing."""
    verdict = validate(_load())
    assert verdict.status == "VALID", verdict.detail
    names = {name for name, _value in verdict.checked}
    assert {"comparator", "threshold table", "float16", "float8_e5m2"} <= names


def test_the_registered_hashes_are_the_files_on_disk():
    """The identity is measured here, not taken from the registry."""
    registry = _load()
    assert _digest(COMPARATOR) == registry["comparator"]["sha256"]
    assert _digest(THRESHOLD_TABLE) == registry["threshold_table"]["sha256"]
    assert registry["comparator"]["repo_head"], "the comparator head is not stated"


def test_literal_eval_cannot_read_these_sources_and_our_fold_can():
    """Pins WHY the folder exists, so it cannot be replaced by literal_eval."""
    source = "2 ** -10"
    with pytest.raises(ValueError):
        ast.literal_eval(source)
    assert fold(source) == 2 ** -10
    assert fold("1.0") == 1.0


def test_the_fold_accepts_the_two_shapes_and_refuses_everything_else():
    assert fold("2 ** -5") == 2 ** -5
    assert fold("1.0") == 1.0
    for bad in ("2 ** 0.5", "True", chr(39) + "x" + chr(39), "2 ** n",
                "float(1)", "2 ** -10 + 1"):
        with pytest.raises(Blocked):
            fold(bad)


@pytest.mark.parametrize("dtype,expected", sorted(EXPECTED.items()))
def test_each_dtype_quintuple_is_the_deployed_one(dtype, expected):
    entry = _load()["threshold_table"]["dtype_entries"][dtype]
    assert tuple(float(value) for value in entry["values"]) == expected
    assert len(entry["source_forms"]) == len(FIELDS)


def test_the_constant_entries_are_not_mistaken_for_exponents():
    """float8_e5m2 last two fields are 1.0, not 2 ** 0 and not a missing value."""
    entry = _load()["threshold_table"]["dtype_entries"]["float8_e5m2"]
    assert entry["values"][-2:] == [1.0, 1.0]
    assert entry["source_forms"][-2:] == ["1.0", "1.0"]
    assert fold(entry["source_forms"][-2]) == 1.0
    assert fold(entry["source_forms"][0]) == 2 ** -2


def test_the_case_distribution_is_the_measured_one():
    registry = _load()
    counts = {}
    for case in registry["cases"]:
        operator = case["operator"]
        if str(case.get("supplemental", "") or "").strip() or \
                "supplemental" in str(case["case_id"]):
            operator = "supplemental_mla"
        key = (operator, case["dtype"])
        counts[key] = counts.get(key, 0) + 1
    assert len(registry["cases"]) == 44
    assert counts[("attention_residual", "bfloat16")] == 17
    assert counts[("attention_residual", "float16")] == 3
    assert counts[("mla", "float16")] == 15
    assert counts[("mla", "bfloat16")] == 5
    assert counts[("supplemental_mla", "float16")] == 2
    assert counts[("supplemental_mla", "bfloat16")] == 2
    assert sum(counts.values()) == 44


@pytest.mark.parametrize("change,needle", [
    ({"comparator.sha256": "0" * 64}, "comparator"),
    ({"threshold_table.sha256": "f" * 64}, "threshold table"),
    ({"comparator.repo_head": "0" * 40}, "comparator"),
    ({"threshold_table.path": "/tmp/elsewhere.py"}, "threshold_table.path"),
    ({"threshold_table.cann_bench_fallback": _DELETE}, "fallback"),
    ({"threshold_table.line_range": [82, 133]}, "fallback"),
    ({"draft": False}, "DRAFT"),
    ({"cases.0.execution_status": "PASS"}, "result"),
    ({"cases.0.registration_basis": ""}, "registration basis"),
])
def test_a_tampered_claim_is_blocked(change, needle):
    with pytest.raises(Blocked) as caught:
        validate(_variant(**change))
    assert caught.value.code == BLOCKED
    assert needle in (caught.value.detail + " " + caught.value.where), caught.value


def test_a_changed_exponent_is_blocked():
    """2 ** -10 rewritten as 2 ** -9 is a different deployment."""
    forms = list(_load()["threshold_table"]["dtype_entries"]["float16"]["source_forms"])
    forms[0] = "2 ** -9"
    with pytest.raises(Blocked) as caught:
        validate(_variant(**{"threshold_table.dtype_entries.float16.source_forms": forms}))
    assert caught.value.code == BLOCKED
    assert "threshold" in caught.value.detail


def test_a_one_point_zero_rewritten_as_an_exponent_is_blocked():
    """The constant branch must not be silently interchangeable with 2 ** 0."""
    forms = list(_load()["threshold_table"]["dtype_entries"]["float8_e5m2"]["source_forms"])
    forms[3] = "2 ** 0"
    assert fold(forms[3]) == 1.0
    with pytest.raises(Blocked) as caught:
        validate(_variant(**{"threshold_table.dtype_entries.float8_e5m2.source_forms": forms}))
    assert caught.value.code == BLOCKED


def test_a_reordered_field_list_is_blocked():
    """Order is meaning: a case binds its entry by position, not by name."""
    reordered = list(FIELDS)
    reordered[1], reordered[3] = reordered[3], reordered[1]
    with pytest.raises(Blocked) as caught:
        validate(_variant(**{"threshold_table.field_order": reordered}))
    assert caught.value.code == BLOCKED
    assert "order" in (caught.value.detail + caught.value.where)


def test_a_reordered_case_entry_is_blocked():
    entry = json.loads(json.dumps(_load()["cases"][0]["threshold_entry"]))
    order = list(entry["order"])
    order[0], order[1] = order[1], order[0]
    entry["order"] = order
    with pytest.raises(Blocked) as caught:
        validate(_variant(**{"cases.0.threshold_entry": entry}))
    assert caught.value.code == BLOCKED


def test_a_case_whose_dtype_has_no_table_entry_is_blocked():
    """A dtype the TABLE carries is still wrong if this case is not that dtype."""
    case = json.loads(json.dumps(_load()["cases"][0]))
    case["dtype"] = "float8_e5m2"
    case["source_key"] = _load()["cases"][0]["source_key"]
    case["threshold_entry"] = {
        "order": list(FIELDS),
        "source_forms": ["2 ** -2", "2 ** -3", "2 ** -5", "1.0", "1.0"],
        "values": list(EXPECTED["float8_e5m2"]),
    }
    registry = _load()
    registry["cases"][0] = case
    with pytest.raises(Blocked) as caught:
        validate(registry)
    assert caught.value.code == BLOCKED
    # a case cannot borrow another case's dtype: the source case states its own,
    # so the refusal comes from the per-case binding before any distribution
    # arithmetic, which is the more precise place for it.
    assert ("dtype" in caught.value.detail
            or "distribution" in caught.value.detail), caught.value.detail


def test_a_missing_field_is_blocked():
    forms = list(_load()["threshold_table"]["dtype_entries"]["float32"]["source_forms"])[:4]
    with pytest.raises(Blocked) as caught:
        validate(_variant(**{"threshold_table.dtype_entries.float32.source_forms": forms}))
    assert caught.value.code == BLOCKED


def test_a_changed_registered_value_is_blocked():
    values = list(_load()["threshold_table"]["dtype_entries"]["bfloat16"]["values"])
    values[1] = values[1] * 2
    with pytest.raises(Blocked) as caught:
        validate(_variant(**{"threshold_table.dtype_entries.bfloat16.values": values}))
    assert caught.value.code == BLOCKED


def test_the_validator_never_writes_the_registry():
    """Read-only is asserted, not promised."""
    before = REGISTRY.read_bytes()
    validate(_load())
    with pytest.raises(Blocked):
        validate(_variant(**{"comparator.sha256": "0" * 64}))
    assert REGISTRY.read_bytes() == before


@pytest.mark.parametrize("field", sorted(REQUIRED_KEYS))
def test_deleting_any_required_top_level_field_is_blocked(field):
    """Shape is a claim: a registry missing a field is not a weaker registry.

    Seven of these used to stay VALID when deleted - among them unknowns (the
    only statement of what is NOT established) and tree_anchor (the only
    in-repository evidence), which is why shape is checked before any value.
    """
    with pytest.raises(Blocked) as caught:
        validate(_variant(**{field: _DELETE}))
    assert caught.value.code == BLOCKED
    assert field in (caught.value.detail + " " + caught.value.where), caught.value


def test_an_emptied_honesty_section_is_blocked():
    """The negatives cannot be hollowed out while the key still exists."""
    for change in ({"claims_not_made": []},
                   {"claims_not_made": ["we are careful"]},
                   {"unknowns": []},
                   {"unknowns": ["nothing"]}):
        with pytest.raises(Blocked) as caught:
            validate(_variant(**change))
        assert caught.value.code == BLOCKED
        # An EMPTY section is caught by the shape guard; a section that still has
        # entries but lost its negations is caught by the honesty guard. Both are
        # refusals of the same claim, so either layer may be the one that fires.
        assert ("honesty" in caught.value.where
                or "top level" in caught.value.where), caught.value


def test_a_tree_anchor_that_no_longer_matches_the_file_is_blocked():
    """The in-repository anchor is measured, not believed."""
    with pytest.raises(Blocked) as caught:
        validate(_variant(**{"tree_anchor.sha256": "0" * 64}))
    assert caught.value.code == BLOCKED
    assert "anchor" in caught.value.where or "anchored" in caught.value.detail


def test_a_tree_anchor_line_that_no_longer_reads_as_registered_is_blocked():
    poisoned = dict(_load()["tree_anchor"]["line_contents"])
    poisoned["983"] = "from some_other_module import compare_case_result"
    with pytest.raises(Blocked) as caught:
        validate(_variant(**{"tree_anchor.line_contents": poisoned}))
    assert caught.value.code == BLOCKED


def test_deleting_one_case_is_named_as_a_case_set_failure():
    """43 of 44 must fail for the RIGHT reason, not as a dtype-count side effect."""
    registry = _load()
    dropped = registry["cases"].pop()
    with pytest.raises(Blocked) as caught:
        validate(registry)
    assert caught.value.code == BLOCKED
    assert "case set" in caught.value.where, caught.value
    # the missing SOURCE KEY is named, which is the anchor a deletion breaks
    assert dropped["source_key"] in caught.value.detail, caught.value.detail


def test_a_dropped_authoritative_case_is_named_as_a_case_set_failure():
    """The ids are bound to case_matrix, so a deletion names the missing id."""
    registry = _load()
    dropped = registry["cases"].pop(0)
    with pytest.raises(Blocked) as caught:
        validate(registry)
    assert caught.value.code == BLOCKED
    assert "case set" in caught.value.where, caught.value
    # the message names the SOURCE key that vanished, which is the global anchor
    assert dropped["source_key"] in caught.value.detail, caught.value.detail


def test_dropping_a_supplemental_case_is_named_as_a_case_set_failure():
    registry = _load()
    # drop exactly ONE supplemental case; the other three stay
    dropped = None
    for index, case in enumerate(registry["cases"]):
        if "supplemental" in str(case.get("source_file", "")):
            dropped = registry["cases"].pop(index)
            break
    assert dropped is not None
    with pytest.raises(Blocked) as caught:
        validate(registry)
    assert caught.value.code == BLOCKED
    assert "case set" in caught.value.where, caught.value
    assert dropped["source_key"] in caught.value.detail, caught.value.detail


def test_a_renamed_source_key_is_blocked():
    """The source key is the anchor: a case cannot carry one no source states."""
    registry = _load()
    for case in registry["cases"]:
        if "supplemental" in str(case.get("source_file", "")):
            case["source_key"] = "supplemental/mla_not_a_source_case"
            break
    with pytest.raises(Blocked) as caught:
        validate(registry)
    assert caught.value.code == BLOCKED
    assert "case set" in caught.value.where


def test_both_key_shapes_are_accepted_and_a_mismatched_marker_is_blocked():
    """40 cases carry no supplemental key and 4 do; one uniform set is wrong."""
    registry = _load()
    without = [case for case in registry["cases"] if "supplemental" not in case]
    with_marker = [case for case in registry["cases"] if "supplemental" in case]
    assert len(without) == 40 and len(with_marker) == 4
    # a supplemental case that loses its marker now disagrees with its own id
    broken = _load()
    for case in broken["cases"]:
        if "supplemental" in case["case_id"]:
            case.pop("supplemental")
            break
    with pytest.raises(Blocked) as caught:
        validate(broken)
    assert caught.value.code == BLOCKED
    assert "case set" in caught.value.where


def test_a_case_naming_a_file_its_key_did_not_come_from_is_blocked():
    """source_file is bound, not decorative: the validator reads that path."""
    registry = _load()
    target = next(case for case in registry["cases"]
                  if case["source_file"].endswith("case_matrix.json"))
    target["source_file"] = "work_package/cases/supplemental_mla.json"
    with pytest.raises(Blocked) as caught:
        validate(registry)
    assert caught.value.code == BLOCKED
    assert "case set" in caught.value.where


def test_a_case_naming_an_unknown_source_file_is_blocked():
    registry = _load()
    registry["cases"][1]["source_file"] = "/tmp/not_a_source_index.json"
    with pytest.raises(Blocked) as caught:
        validate(registry)
    assert caught.value.code == BLOCKED
    assert "case set" in caught.value.where


def test_dropping_an_authoritative_case_leaves_the_source_case_id_set_intact():
    """WHY source_key is the only global anchor, measured rather than argued.

    The authoritative case ids repeat across operators, so a deleted case leaves
    the source_case_id multiset satisfiable while the source_key set is not. That
    is the executable reason the global binding uses the key.
    """
    registry = _load()
    dropped = None
    for index, case in enumerate(registry["cases"]):
        if case["source_file"].endswith("case_matrix.json"):
            dropped = registry["cases"].pop(index)
            break
    assert dropped is not None
    source = _source_indices()
    remaining_ids = sorted(str(case["source_case_id"]) for case in registry["cases"])
    expected_ids = sorted(str(record["case_id"]) for record in source.values())
    assert remaining_ids != expected_ids, (
        "the id multiset must not cover the loss; if it did, the id would be a "
        "global anchor and the key would not be needed")
    with pytest.raises(Blocked) as caught:
        validate(registry)
    assert caught.value.code == BLOCKED
    assert dropped["source_key"] in caught.value.detail

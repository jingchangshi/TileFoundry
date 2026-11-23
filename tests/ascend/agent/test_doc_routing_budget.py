"""Anti-drift budget and section-sliced routing verification for Ascend architecture docs."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"
_CLAUDE_PATH = _REPO_ROOT / "CLAUDE.md"
_DOCS_DIR = _REPO_ROOT / "docs" / "ascend"

_AUDIENCE_BANNER = (
    "> Audience: Human / First-Time Architect Overview. "
    "Coding subtasks DO NOT read this file; follow section-sliced routes in AGENTS.md."
)

_EXPECTED_SECTIONS: dict[str, tuple[str, ...]] = {
    "01_mece_contracts_and_routing.md": (
        "## [SEC-C1] Problem Space & Bounded 2-4 Regimes",
        "## [SEC-C2] Action Lattice, 3-Layer Prompt & COW Isolation",
        "## [SEC-C3] 5D Observation, 13 Symptom Families & Frozen KB",
    ),
    "02_residual_convergence_and_gates.md": (
        "## [SEC-G1] 3-Tier Cost Model Sink-Down & Sample-Gated Basis",
        "## [SEC-G2] Unified Compile Session & Anchor-Specialized Two-Snapshot Ledger",
        "## [SEC-G3] Compilation Idempotence & Orthogonal Tail Guard",
    ),
}

_REQUIRED_SEC_IDS = frozenset(
    {"SEC-C1", "SEC-C2", "SEC-C3", "SEC-G1", "SEC-G2", "SEC-G3"}
)

_ROUTE_PATTERN = re.compile(
    r"\|\s*[^|\n]+\|\s*`\[(SEC-[CG]\d+)\]`\s*\|\s*"
    r"\[[^\]]+\(`L(\d+)-L(\d+)`\)\]\((\./docs/ascend/[^)#]+)#L(\d+)-L(\d+)\)\s*\|"
)


def _parse_agents_routes(agents_text: str) -> dict[str, tuple[str, int, int]]:
    """Extract all declared ``[SEC-*]`` section-sliced routes from ``AGENTS.md``."""
    routes: dict[str, tuple[str, int, int]] = {}
    for match in _ROUTE_PATTERN.finditer(agents_text):
        sec_id = match.group(1)
        label_start = int(match.group(2))
        label_end = int(match.group(3))
        rel_path = match.group(4)
        anchor_start = int(match.group(5))
        anchor_end = int(match.group(6))
        assert label_start == anchor_start, (
            f"{sec_id} label start {label_start} != anchor start {anchor_start}"
        )
        assert label_end == anchor_end, (
            f"{sec_id} label end {label_end} != anchor end {anchor_end}"
        )
        assert sec_id not in routes, f"duplicate route for {sec_id} in AGENTS.md"
        routes[sec_id] = (rel_path, anchor_start, anchor_end)
    return routes


def test_claude_md_points_strictly_to_agents() -> None:
    """Ensure ``CLAUDE.md`` remains an exact ``@AGENTS.md`` pointer."""
    raw = _CLAUDE_PATH.read_text(encoding="utf-8")
    assert raw.strip() == "@AGENTS.md"
    assert raw == "@AGENTS.md\n"


def test_agents_md_budget_and_routing_discipline() -> None:
    """Enforce hard line/byte caps and clean relative routing rules on ``AGENTS.md``."""
    raw_bytes = _AGENTS_PATH.read_bytes()
    text = raw_bytes.decode("utf-8")
    lines = text.splitlines()
    assert len(lines) <= 60, f"AGENTS.md has {len(lines)} lines (max 60)"
    assert len(raw_bytes) <= 4800, f"AGENTS.md is {len(raw_bytes)} bytes (max 4800)"
    assert "00_architecture_overview.md" in text
    assert "DO NOT" in text
    assert "NEVER" in text
    assert re.search(r"\bV\d+\b", text) is None
    assert "§" not in text


def test_architecture_docs_line_budgets_and_exact_headings() -> None:
    """Enforce line budgets, audience banner, and exact ``##`` anchors in ``docs/ascend``."""
    overview_lines = (_DOCS_DIR / "00_architecture_overview.md").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(overview_lines) <= 140, f"overview has {len(overview_lines)} lines"
    assert overview_lines[0] == _AUDIENCE_BANNER

    for filename, expected_headings in _EXPECTED_SECTIONS.items():
        doc_lines = (_DOCS_DIR / filename).read_text(encoding="utf-8").splitlines()
        assert len(doc_lines) <= 170, f"{filename} has {len(doc_lines)} lines"
        h2_headings = tuple(line for line in doc_lines if line.startswith("## "))
        assert h2_headings == expected_headings, (
            f"{filename} H2 headings {h2_headings!r} != {expected_headings!r}"
        )


def test_agents_section_sliced_routes_match_exact_line_ranges() -> None:
    """Verify every ``[SEC-*]`` route in ``AGENTS.md`` resolves to exact heading bounds."""
    agents_text = _AGENTS_PATH.read_text(encoding="utf-8")
    routes = _parse_agents_routes(agents_text)
    assert set(routes) == _REQUIRED_SEC_IDS

    lines_by_rel: dict[str, list[str]] = {}
    sec_starts_by_rel: dict[str, list[int]] = {}
    for rel_path, _, _ in routes.values():
        if rel_path in lines_by_rel:
            continue
        target = (_REPO_ROOT / rel_path).resolve()
        assert target.is_file(), f"missing routed file {rel_path}"
        lines = target.read_text(encoding="utf-8").splitlines()
        lines_by_rel[rel_path] = lines
        sec_starts_by_rel[rel_path] = [
            idx for idx, line in enumerate(lines, start=1) if line.startswith("## [SEC-")
        ]

    for sec_id, (rel_path, start, end) in routes.items():
        lines = lines_by_rel[rel_path]
        assert 1 <= start <= end <= len(lines), (
            f"{sec_id} range L{start}-L{end} out of bounds for {rel_path}"
        )
        assert lines[start - 1].startswith(f"## [{sec_id}]"), (
            f"{rel_path}:{start} does not start with '## [{sec_id}]'"
        )
        starts = sec_starts_by_rel[rel_path]
        pos = starts.index(start)
        expected_end = starts[pos + 1] - 1 if pos + 1 < len(starts) else len(lines)
        assert end == expected_end, (
            f"{sec_id} end L{end} != expected L{expected_end} in {rel_path}"
        )

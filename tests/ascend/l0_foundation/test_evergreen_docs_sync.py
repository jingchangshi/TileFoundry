"""Tests for evergreen Ascend governance docs, byte budget, and section anchors."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAX_AGENTS_BYTES = 4800
_DOC_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    "docs/ascend/00_architecture_overview.md",
    "docs/ascend/01_mece_contracts_and_routing.md",
    "docs/ascend/02_residual_convergence_and_gates.md",
    "targets/ascend/README.md",
)
_OBSOLETE_DOC_SYMBOLS: tuple[str, ...] = (
    "msprof_report_parser.py",
    "sync_dav3510_cann_surface.py",
    "cann_9_2",
    "surrogate_cannbench.py",
)
_EVERGREEN_LAYER_MARKERS: tuple[tuple[str, str], ...] = (
    ("AGENTS.md", "agent/gates/"),
    ("AGENTS.md", "tests/ascend/l0_foundation/"),
    ("AGENTS.md", "regime_slice_patcher"),
    ("AGENTS.md", "seed_synthesizer"),
    ("targets/ascend/README.md", "L0 < L1 < L2 < (L3 || L4) < L5"),
    ("docs/ascend/00_architecture_overview.md", "L0 < L1 < L2 < (L3 || L4) < L5"),
)
_SEC_ROW_RE = re.compile(
    r"\|\s*`(\[SEC-[A-Z0-9]+\])`\s*\|\s*\[`[^`]+` \(`L(\d+)-L(\d+)`\)\]"
    r"\(\./(docs/ascend/[^#)]+)#L(\d+)-L(\d+)\)\s*\|"
)


class DocSourcePort(Protocol):
    """Read-only port providing repository doc text and byte size."""

    def read_text(self, rel_path: str) -> str:
        """Return UTF-8 text for *rel_path*."""

    def byte_size(self, rel_path: str) -> int:
        """Return UTF-8 byte length for *rel_path*."""


class RepoDocSource:
    """Repository filesystem implementation of :class:`DocSourcePort`."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def read_text(self, rel_path: str) -> str:
        """Read UTF-8 text from *rel_path* under repository root."""
        return (self._root / rel_path).read_text(encoding="utf-8")

    def byte_size(self, rel_path: str) -> int:
        """Return raw byte count of *rel_path* under repository root."""
        return len((self._root / rel_path).read_bytes())


class FakeDocSource:
    """In-memory :class:`DocSourcePort` for reject/accept validation tests."""

    def __init__(self, docs: dict[str, str]) -> None:
        self._docs = dict(docs)

    def read_text(self, rel_path: str) -> str:
        """Return stored text for *rel_path*."""
        return self._docs[rel_path]

    def byte_size(self, rel_path: str) -> int:
        """Return UTF-8 byte length of stored text for *rel_path*."""
        return len(self._docs[rel_path].encode("utf-8"))


class EvergreenDocValidator:
    """Validate evergreen Ascend doc budgets, symbols, and ``[SEC-*]`` line anchors."""

    def __init__(self, doc_port: DocSourcePort) -> None:
        self._port = doc_port

    def validate(self) -> tuple[str, ...]:
        """Return validation failure messages; empty tuple means accepted."""
        errors: list[str] = []
        if (agents_bytes := self._port.byte_size("AGENTS.md")) > _MAX_AGENTS_BYTES:
            errors.append(f"AGENTS.md exceeds {_MAX_AGENTS_BYTES} bytes: {agents_bytes}")
        if self._port.read_text("CLAUDE.md").strip() != "@AGENTS.md":
            errors.append("CLAUDE.md must remain @AGENTS.md")
        for doc_path in _DOC_PATHS:
            text = self._port.read_text(doc_path)
            for sym in _OBSOLETE_DOC_SYMBOLS:
                if sym in text:
                    errors.append(f"{doc_path} contains obsolete symbol {sym!r}")
        for doc_path, marker in _EVERGREEN_LAYER_MARKERS:
            if marker not in self._port.read_text(doc_path):
                errors.append(f"{doc_path} missing evergreen marker {marker!r}")
        agents_text = self._port.read_text("AGENTS.md")
        matches = list(_SEC_ROW_RE.finditer(agents_text))
        if len(matches) < 6:
            errors.append(f"AGENTS.md expected >= 6 [SEC-*] routes, found {len(matches)}")
        for m in matches:
            sec_id, d_s, d_e, target_rel, l_s, l_e = m.groups()
            s_num, e_num = int(l_s), int(l_e)
            if (int(d_s), int(d_e)) != (s_num, e_num):
                errors.append(f"{sec_id} label L{d_s}-L{d_e} != anchor #L{l_s}-L{l_e}")
            target_lines = self._port.read_text(target_rel).splitlines()
            if not (1 <= s_num <= e_num <= len(target_lines)):
                errors.append(f"{sec_id} anchor L{s_num}-L{e_num} out of range for {target_rel}")
                continue
            if not target_lines[s_num - 1].startswith(f"## {sec_id} "):
                errors.append(f"{sec_id} start line L{s_num} does not begin with '## {sec_id} '")
        return tuple(errors)


def _baseline_docs_map() -> dict[str, str]:
    docs = {p: (_REPO_ROOT / p).read_text(encoding="utf-8") for p in _DOC_PATHS}
    docs["CLAUDE.md"] = (_REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    return docs


@pytest.mark.parametrize(
    ("target_doc", "mutation_kind", "needle"),
    [
        ("AGENTS.md", "append_obsolete", "msprof_report_parser.py"),
        ("targets/ascend/README.md", "append_obsolete", "sync_dav3510_cann_surface.py"),
        ("docs/ascend/01_mece_contracts_and_routing.md", "append_obsolete", "cann_9_2"),
        ("docs/ascend/02_residual_convergence_and_gates.md", "append_obsolete",
         "surrogate_cannbench.py"),
        ("AGENTS.md", "oversize_agents", "AGENTS.md exceeds"),
        ("AGENTS.md", "bad_sec_anchor", "out of range"),
    ],
)
def test_evergreen_docs_rejects_obsolete_symbols_size_and_bad_anchors(
    target_doc: str,
    mutation_kind: str,
    needle: str,
) -> None:
    """Reject obsolete doc symbols, AGENTS.md budget overflow, and broken [SEC-*] ranges."""
    docs = _baseline_docs_map()
    if mutation_kind == "append_obsolete":
        docs[target_doc] = f"{docs[target_doc]}\n{needle}\n"
    elif mutation_kind == "oversize_agents":
        docs[target_doc] = f"{docs[target_doc]}\n{'x' * 300}\n"
    elif mutation_kind == "bad_sec_anchor":
        docs[target_doc] = docs[target_doc].replace("L5-L21", "L5-L999")
    validator = EvergreenDocValidator(FakeDocSource(docs))
    errors = validator.validate()
    assert errors
    assert any(needle in err for err in errors)


def test_evergreen_docs_accepts_repo_contracts_and_sec_anchors() -> None:
    """Accept synchronized repo docs with AGENTS.md <= 4800 B and exact [SEC-*] ranges."""
    port = RepoDocSource(_REPO_ROOT)
    validator = EvergreenDocValidator(port)
    assert port.byte_size("AGENTS.md") <= _MAX_AGENTS_BYTES
    assert validator.validate() == ()

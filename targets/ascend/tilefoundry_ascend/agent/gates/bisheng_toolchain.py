"""CANN dav-3510 bisheng toolchain discovery, objdump telemetry, and AscendC C++20 compiler."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tilefoundry_ascend.agent.contracts import CompileTaskSpec
from tilefoundry_ascend.ascend.compile_cache.artifact_store import (
    ArtifactStore, compute_compile_task_key, is_verified_binary_file, resolve_unified_tl_cache_dir,
)
from tilefoundry_ascend.ascend.compile_cache.cpp_closure import (
    extract_reachable_trackb_source, strip_cpp_comments_and_strings,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    BishengResourceSummary, parse_bisheng_res_usage,
)
from tilefoundry_ascend.ascend.cost_model.microarch_profile import extract_microarch_profile_from_cce
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_COMPILE_DIAGNOSTIC_LIMIT = 1500
_CANN_INCLUDE_SUBDIRS: tuple[str, ...] = (
    "include", "x86_64-linux/asc", "x86_64-linux/asc/include", "x86_64-linux/include",
    "aarch64-linux/asc", "aarch64-linux/asc/include", "aarch64-linux/include",
    "runtime/include", "asc", "asc/include", "tools/tikcpp/tikcfw", "compiler/tikcpp/tikcfw",
)
_VLOOPV2_MISS_RE = re.compile(
    r"(?:vloop_?v2\s*(?:miss|failed|fallback|disabled|not\s+applied)|vloop_?v2\s*[:=]\s*(?:false|0|no|miss)|cannot\s+use\s+vloop_?v2)",
    re.IGNORECASE,
)
_COMPILE_ERROR_RE = re.compile(
    r"(?i)(?:\[error\]|\btvm\s*error\b|\bfatal\s*:|\bCheck\s+failed:|(?:^|[\s:(])(?:Assertion|Type|Value|Runtime|Internal|Key|Attribute|Name|Syntax)Error:|(?:^|[\s:\[(])(?:fatal\s+)?error:)"
)
_DEPRECATED_WARN_RE = re.compile(
    r"(?:warning:\s+.*\[-Wdeprecated-declarations\]|note:\s+'[^']+'\s+has been explicitly marked deprecated here|note:\s+expanded from macro 'ASC_DEPRECATED')"
)
_NON_ERROR_DIAG_HEADER_RE = re.compile(r"^(?:note:|warning:|in file included from|/\S+:\d+:\d+:\s*(?:warning|note):)", re.IGNORECASE)
_TVM_BT_RE = re.compile(r"^\s*(?:\[bt\]\s*\(\d+\)|File \"<unknown>\", line 0, in )")
_OVERFLOW_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("VF_STACK_OVERFLOW", ("exceeded vf stack size", "vf_stack", "vf stack", "stack frame size", "stack overflow")),
    ("L0C_SRAM_OVERFLOW", ("l0c_sram_overflow", "l0c overflow", "l0c_overflow", "l0c capacity", "exceeds l0c", "exceeded l0c")),
    ("L1_SRAM_OVERFLOW", ("l1_sram_overflow", "l1 overflow", "l1_overflow", "cbuf overflow", "l1 capacity", "cbuf capacity", "exceeds l1", "exceeded l1")),
    ("UB_SRAM_OVERFLOW", ("ub_sram_overflow", "ub_overflow", "ubuf_overflow", "ub overflow", "ubuf overflow", "ub capacity", "ubuf capacity", "exceeds ub", "exceeded ub", "exceeds ubuf", "exceeded ubuf")),
)
_OBJDUMP_ZERO_KEYS = (
    "total_insns", "cube_mmad", "fixpipe", "mte1", "mte2", "mte3", "vec", "vloopv2",
    "dcci", "sync", "sw_back_branches", "vf_sw_back_branches", "scalar_regs", "vec_regs",
    "gm_to_l1_insns", "gm_to_ub_insns", "l1_to_l0a_insns", "l1_to_l0b_insns",
    "l0c_to_ub_insns", "l0c_to_l1_insns", "l0c_to_gm_insns", "ub_to_gm_insns", "ub_to_l1_insns",
)


class SubprocResultLike(Protocol):
    """Completed subprocess result surface."""

    returncode: int
    stdout: str | None
    stderr: str | None


@dataclass(frozen=True)
class StageBResult:
    """Output of Stage B Conda bisheng compilation check."""

    passed: bool
    error_message: str = ""
    elapsed_sec: float = 0.0
    spill_bytes: int = 0
    vloopv2_hit: bool = True
    compiler_summary: BishengResourceSummary | None = None
    microarch_profile: dict[str, object] | None = None
    full_diagnostic: str = ""
    generated_cce: str = ""
    lowered_tir: str = ""
    fault_plane: str = ""
    fault_code: str = ""
    failed_tier: str = ""
    artifact_id: str = ""
    binary_paths: tuple[str, ...] = ()


def _stage_b_fail(
    t0: float,
    error_message: str,
    fault_code: str,
    *,
    fault_plane: str = "STAGE_B_COMPILE",
    spill_bytes: int = 0,
    vloopv2_hit: bool = True,
    summary: BishengResourceSummary | None = None,
    ma_profile: dict[str, object] | None = None,
    full_diag: str = "",
    cce: str = "",
    tir: str = "",
) -> StageBResult:
    return StageBResult(
        passed=False, error_message=error_message, elapsed_sec=time.monotonic() - t0,
        spill_bytes=spill_bytes, vloopv2_hit=vloopv2_hit, compiler_summary=summary,
        microarch_profile=ma_profile, full_diagnostic=full_diag, generated_cce=cce,
        lowered_tir=tir, fault_plane=fault_plane, fault_code=fault_code, failed_tier="stage_b",
    )


def has_unreserved_vf_stack_overflow(spec: CompileTaskSpec, stack_bytes: int, compiler_diag: str) -> bool:
    """Decide VF stack overflow strictly against this specialization's flags."""
    low = (compiler_diag or "").lower()
    if "exceeded vf stack size" in low or "stack overflow" in low:
        return True
    return int(stack_bytes) > 0 and spec.disables_vf_stack_ubuf


def _normalize_flag_list(raw_flags: object) -> list[str]:
    if raw_flags is None:
        return []
    if isinstance(raw_flags, str):
        return raw_flags.strip().split() if raw_flags.strip() else []
    if isinstance(raw_flags, (tuple, list)) or (hasattr(raw_flags, "__iter__") and not isinstance(raw_flags, (bytes, bytearray, Mapping))):
        return list(dict.fromkeys(v for item in raw_flags if (v := str(getattr(item, "value", item)).strip())))
    return [s] if (s := str(raw_flags).strip()) else []


def resolve_trackb_compile_flags(cpp_source: str, selector_flags: Sequence[str] | str | None = None) -> tuple[str, ...]:
    """Resolve Track-B bisheng flags, stripping SIMD-VF UBUF flags on __simt_vf__."""
    normalized = _normalize_flag_list(selector_flags)
    if re.search(r"\b__simt_vf__\b", strip_cpp_comments_and_strings(cpp_source)):
        simdvf = frozenset(PURE_SIMDVF_UBUF_FLAGS)
        return tuple(f for f in normalized if f not in simdvf)
    return tuple(dict.fromkeys([*PURE_SIMDVF_UBUF_FLAGS, *normalized]))


def _normalize_trackb_unit_source(cpp_source: str, *, regime_tag: str = "") -> str:
    stripped = cpp_source.strip()
    if regime_tag and stripped:
        masked = strip_cpp_comments_and_strings(stripped)
        if len(re.findall(r"\b__global__\b", masked)) > 1 and "<<<" not in masked:
            _, extracted = extract_reachable_trackb_source(stripped, regime_tag=regime_tag)
            return extracted.strip()
    return stripped


def _bounded_compile_diagnostic(output_text: str, *, limit: int = _COMPILE_DIAGNOSTIC_LIMIT) -> str:
    """Keep primary compiler errors without deprecation cascades plus a bounded output tail."""
    text = output_text.strip()
    if len(text) <= limit:
        return text
    raw_lines = text.splitlines()
    kept: list[str] = []
    skip_cascade = skip_bt = False
    for line in raw_lines:
        if _COMPILE_ERROR_RE.search(line):
            skip_cascade = skip_bt = False
            kept.append(line)
        elif _DEPRECATED_WARN_RE.search(line):
            skip_cascade, skip_bt = True, False
        elif line.strip() == "Stack trace:":
            skip_bt, skip_cascade = True, False
        elif skip_bt:
            if not (_TVM_BT_RE.search(line) or not line.strip()):
                skip_bt = False
                kept.append(line)
        elif skip_cascade:
            st = line.strip()
            if not st or _NON_ERROR_DIAG_HEADER_RE.match(st):
                skip_cascade = False
                if st:
                    kept.append(line)
        else:
            kept.append(line)
    lines = kept if (kept and any(_COMPILE_ERROR_RE.search(l) for l in kept)) else raw_lines
    eff = "\n".join(lines).strip()
    if len(eff) <= limit:
        return eff
    if not (err_idxs := [i for i, ln in enumerate(lines) if _COMPILE_ERROR_RE.search(ln)]):
        return eff[-limit:]
    ei, tail = err_idxs[0], eff[-limit:]
    if lines[ei].strip() in tail:
        return tail
    ctx = "\n".join(lines[ei : min(len(lines), ei + 7)]).strip()
    head = ctx if len(ctx) <= limit // 2 else lines[ei].strip()[: limit // 2]
    prefix = f"{head}\n[... compiler output truncated ...]\n"
    return prefix[:limit] if len(prefix) >= limit else prefix + eff[-(limit - len(prefix)) :]


def _classify_stage_b_overflow_fault_code(combined_diag: str, summary: BishengResourceSummary | None, default_code: str) -> str:
    if summary is not None and int(summary.stack_bytes) > 0:
        return "VF_STACK_OVERFLOW"
    low = (combined_diag or "").lower()
    for code, toks in _OVERFLOW_TABLES:
        if any(t in low for t in toks):
            return code
    if any(c in low for c in ("overflow", "exceed", "capacity", "out of memory", "insufficient")):
        if "l0c" in low or "__cc__" in low:
            return "L0C_SRAM_OVERFLOW"
        if any(t in low for t in ("l1 ", "l1_", "cbuf", "__cbuf__")):
            return "L1_SRAM_OVERFLOW"
        if "ubuf" in low or "__ubuf__" in low or re.search(r"\bub\b", low):
            return "UB_SRAM_OVERFLOW"
    return default_code


def _parse_cce_res_usage_with_summary(output_text: str, *, asm_or_dump_text: str = "") -> tuple[int, bool, BishengResourceSummary]:
    summary = parse_bisheng_res_usage(output_text or "", asm_or_dump_text=asm_or_dump_text or "")
    vloop_hit = summary.vloopv2_loops > 0 and summary.software_loops == 0 and not _VLOOPV2_MISS_RE.search(f"{output_text}\n{asm_or_dump_text}")
    return int(summary.spill_bytes), vloop_hit, summary


def parse_cce_res_usage(output_text: str) -> tuple[int, bool]:
    """Parse bisheng --cce-res-usage diagnostics into (spill_bytes, vloopv2_hit)."""
    sp, vl, _ = _parse_cce_res_usage_with_summary(output_text)
    return sp, vl


def _resolve_cann_root(cann_root: str | Path | None = None, *, bisheng_bin: str | Path | None = None) -> Path | None:
    if cann_root is not None and str(cann_root).strip():
        return Path(cann_root).expanduser()
    for ek in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "BISHENG_HOME"):
        if raw := os.environ.get(ek, "").strip():
            cand = Path(raw).expanduser()
            if ek == "BISHENG_HOME" and not any((cand / s).is_dir() for s in _CANN_INCLUDE_SUBDIRS[:4]):
                for p in cand.parents:
                    if any((p / s).is_dir() for s in _CANN_INCLUDE_SUBDIRS[:4]):
                        return p
            return cand
    for k in ("ASCEND_OPP_PATH", "ASCEND_AICPU_PATH"):
        if (raw := os.environ.get(k, "").strip()) and any(((p := Path(raw).expanduser().parent) / s).is_dir() for s in _CANN_INCLUDE_SUBDIRS[:4]):
            return p
    if rbin := (bisheng_bin or shutil.which("bisheng")):
        for p in Path(str(rbin)).resolve().parents:
            if any((p / s).is_dir() for s in _CANN_INCLUDE_SUBDIRS[:4]):
                return p
    return None


def _resolve_bisheng_bin(bisheng_bin: str | Path | None = None, cann_root: str | Path | None = None) -> str | None:
    if bisheng_bin is not None:
        raw = str(bisheng_bin).strip()
        return (str(Path(raw)) if Path(raw).is_file() else shutil.which(raw)) if raw else None
    if (bh := os.environ.get("BISHENG_HOME", "").strip()) and (cand := Path(bh).expanduser() / "bin" / "bisheng").is_file():
        return str(cand)
    if found := shutil.which("bisheng"):
        return found
    if (rc := _resolve_cann_root(cann_root)) is not None:
        return next((str(rc / rel) for rel in ("tools/ccec_compiler/bin/bisheng", "bin/bisheng") if (rc / rel).is_file()), None)
    return None


def _detect_cann_include_dirs(cann_root: str | Path | None = None, *, bisheng_bin: str | Path | None = None) -> list[str]:
    roots: list[Path] = [Path(str(cann_root)).expanduser()] if (cann_root is not None and str(cann_root).strip()) else []
    roots.extend(Path(r).expanduser() for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if (r := os.environ.get(k, "").strip()))
    if inf := _resolve_cann_root(cann_root, bisheng_bin=bisheng_bin):
        roots.append(inf)
    found, seen = [], set()
    for root in roots:
        for sub in _CANN_INCLUDE_SUBDIRS:
            if (inc := root / sub).is_dir() and (rk := str(inc.resolve())) not in seen:
                seen.add(rk)
                found.append(str(inc))
    return found


def _inject_cann_cpath_env(env_map: dict[str, str] | None = None) -> list[str]:
    if incs := _detect_cann_include_dirs():
        tgt = os.environ if env_map is None else env_map
        tgt["CPATH"] = os.pathsep.join(dict.fromkeys([*incs, *(p for p in tgt.get("CPATH", "").split(os.pathsep) if p)]))
    return incs


def _resolve_toolchain_tool(tool_names: Sequence[str], *, bisheng_bin: str | Path | None = None) -> str | None:
    if bisheng_bin is not None and (bp := Path(str(bisheng_bin))).is_file():
        if hit := next((str(c) for n in tool_names if (c := bp.resolve().with_name(n)).is_file()), None):
            return hit
    return next((w for n in tool_names if (w := shutil.which(n))), None)


def _verify_object_has_instantiated_kernel(
    obj_path: Path, *, bisheng_bin: str | Path | None = None, subproc_run: Callable[..., SubprocResultLike] | None = None
) -> tuple[bool, str]:
    if not obj_path.is_file():
        return False, "stage_b_ascendc_missing_object: object file was not generated"
    run_fn = subproc_run or subprocess.run
    if nm_bin := _resolve_toolchain_tool(("llvm-nm", "nm"), bisheng_bin=bisheng_bin):
        if (proc := run_fn([nm_bin, "--defined-only", str(obj_path)], text=True, capture_output=True, timeout=15, check=False)).returncode == 0:
            if any(re.search(r"\b[Tt]\s+(?!\$)\S+", ln.strip()) for ln in (proc.stdout or "").splitlines()):
                return True, ""
            return False, "stage_b_ascendc_uninstantiated_template: object file contains zero instantiated code symbols (.text is empty); provide concrete template instantiation via _render_trackb_tu"
    if od_bin := _resolve_toolchain_tool(("llvm-objdump", "objdump"), bisheng_bin=bisheng_bin):
        if (proc := run_fn([od_bin, "-h", str(obj_path)], text=True, capture_output=True, timeout=15, check=False)).returncode == 0:
            m = re.search(r"\.text(?:\.\S+)?\s+([0-9a-fA-F]+)\s", proc.stdout or "")
            return (True, "") if (m and int(m.group(1), 16) > 0) else (False, "stage_b_ascendc_uninstantiated_template: .text section size is 0; provide concrete template instantiation via _render_trackb_tu")
    return (True, "") if obj_path.stat().st_size > 2048 else (False, "stage_b_ascendc_uninstantiated_template: object file contains no verifiable instantiated kernel code")


def _disassemble_object(
    obj_path: Path, *, bisheng_bin: str | Path | None = None, subproc_run: Callable[..., SubprocResultLike] | None = None
) -> str:
    od_bin = _resolve_toolchain_tool(("llvm-objdump", "objdump"), bisheng_bin=bisheng_bin)
    if not od_bin or not obj_path.is_file():
        return ""
    proc = (subproc_run or subprocess.run)([od_bin, "-d", str(obj_path)], text=True, capture_output=True, timeout=30, check=False)
    return (proc.stdout or "") if proc.returncode == 0 else ""


def _parse_objdump_pipe_counts(objdump_text: str) -> dict[str, int]:
    counts = dict.fromkeys(_OBJDUMP_ZERO_KEYS, 0)
    if not objdump_text.strip():
        return counts
    xregs, vregs, in_vf = set(), set(), False
    for raw_ln in objdump_text.splitlines():
        if lbl := re.match(r"^[0-9a-f]+\s+<([^>]+)>:\s*$", raw_ln.strip()):
            in_vf = ".vector.thread" in lbl.group(1)
            continue
        if not (m := re.match(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+([A-Za-z0-9_.]+)(.*)$", raw_ln)):
            continue
        pc, mnem, rest = int(m.group(1), 16), m.group(2).upper(), m.group(3)
        counts["total_insns"] += 1
        xregs.update(int(r) for r in re.findall(r"\bX(\d+)\b", rest))
        vregs.update(int(r) for r in re.findall(r"\bV(\d+)\b", rest))
        if mnem.startswith("VLOOP"):
            counts["vloopv2"] += int("VLOOPV2" in mnem or "VLOOP_V2" in mnem)
        elif mnem.startswith("MMAD"):
            counts["cube_mmad"] += 1
        elif mnem.startswith("FIX_"):
            counts["fixpipe"] += 1
            for tok, key in (("L1", "l0c_to_l1_insns"), ("CBUF", "l0c_to_l1_insns"), ("UB", "l0c_to_ub_insns"), ("GM", "l0c_to_gm_insns")):
                if tok in mnem:
                    counts[key] += 1
                    break
        elif mnem.startswith("LOAD_L1_TO_L0"):
            counts["mte1"] += 1
            counts["l1_to_l0b_insns" if "L0B" in mnem else "l1_to_l0a_insns"] += 1
        elif mnem.startswith("MOV_OUT_TO_"):
            counts["mte2"] += 1
            if "L1" in mnem or "CBUF" in mnem:
                counts["gm_to_l1_insns"] += 1
            elif "UB" in mnem:
                counts["gm_to_ub_insns"] += 1
        elif mnem.startswith("MOV_UB_TO_"):
            counts["mte3"] += 1
            counts["ub_to_l1_insns" if ("L1" in mnem or "CBUF" in mnem) else "ub_to_gm_insns"] += 1
        elif mnem.startswith("DCCI"):
            counts["dcci"] += 1
        elif mnem.startswith(("SET_FLAG", "WAIT_FLAG", "SET_INTRA", "WAIT_INTRA", "BAR.")):
            counts["sync"] += 1
        elif mnem.startswith("V") and not mnem.startswith("VF"):
            counts["vec"] += 1
        if mnem.startswith(("JUMP", "B.", "CBNZ", "CBZ", "TBNZ", "TBZ")) or mnem == "B":
            tm, hm = re.search(r"\+0x([0-9a-f]+)>", rest, re.IGNORECASE), re.search(r"#(-?\d+)", rest)
            if (tm and int(tm.group(1), 16) <= pc) or (not tm and hm and int(hm.group(1)) <= pc):
                counts["sw_back_branches"] += 1
                counts["vf_sw_back_branches"] += int(in_vf)
    counts["scalar_regs"], counts["vec_regs"] = len(xregs), len(vregs)
    return counts


def _augment_bisheng_telemetry_text(compiler_out: str, objdump_text: str) -> str:
    dis = _parse_objdump_pipe_counts(objdump_text)
    vec_regs = max([*(int(m) for m in re.findall(r"\bVReg\s+number\s*:\s*(\d+)", compiler_out)), dis["vec_regs"], 0])
    stack_b = max((int(m) for m in re.findall(r"(?:^|[,\s])Stack\s+size\s*:\s*(\d+)\s*bytes", compiler_out, re.IGNORECASE)), default=0)
    extra: list[str] = []
    if vec_regs > 0 and not re.search(r"(?:vector\s+register\s+usage|vec_reg_num)", compiler_out, re.IGNORECASE):
        extra.append(f"Vector Register Usage: {vec_regs}")
    if dis["scalar_regs"] > 0 and not re.search(r"(?:scalar\s+register\s+usage|scalar_reg_num)", compiler_out, re.IGNORECASE):
        extra.append(f"Scalar Register Usage: {dis['scalar_regs']}")
    if stack_b > 0 and not re.search(r"stack\s+frame\s+size", compiler_out, re.IGNORECASE):
        extra.append(f"Stack Frame Size: {stack_b} bytes")
    if objdump_text.strip():
        vloop_cnt = dis["vloopv2"]
        extra.extend((f"vloopv2_loops: {vloop_cnt}", f"software_loops: {dis['vf_sw_back_branches'] if vloop_cnt > 0 else dis['sw_back_branches']}"))
    return f"{compiler_out}\n" + "\n".join(extra) if extra else compiler_out


def _detect_tilelang_include_dir() -> str | None:
    try:
        import tilelang  # noqa: PLC0415

        if ((sd := Path(tilelang.__file__).resolve().parents[1] / "src") / "tl_templates" / "ascend").is_dir():
            return str(sd)
    except Exception:
        pass
    for raw in (os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE", "").strip(), os.environ.get("TILEFOUNDRY_NPU_TILELANG_REPO", "").strip()):
        if raw and ((sd := Path(raw).expanduser() / "src") / "tl_templates" / "ascend").is_dir():
            return str(sd)
    sib = Path(__file__).resolve().parents[5].parent / "tilelang" / "src"
    return str(sib) if (sib / "tl_templates" / "ascend").is_dir() else None


@dataclass(frozen=True)
class _AscendcCompileContext:
    extra_flags: list[str]
    has_explicit_x: bool
    std_flags: list[str]
    include_flags: list[str]
    env: dict[str, str]


def _prepare_ascendc_compile_context(
    *, compile_flags: Sequence[str] | str | None, cann_root: str | Path | None,
    resolved_bisheng: str, extra_include_dirs: Sequence[str | Path] | str | Path | None,
) -> _AscendcCompileContext:
    extra_flags = _normalize_flag_list(compile_flags)
    std_flags = [] if any(f.startswith("-std=") for f in extra_flags) else ["-std=c++20"]
    include_flags, seen = [], set(extra_flags)
    raw_incs: list[str] = [str(extra_include_dirs)] if isinstance(extra_include_dirs, (str, Path)) else [str(i) for i in (extra_include_dirs or ())]
    if tl_inc := _detect_tilelang_include_dir():
        raw_incs.append(tl_inc)
    raw_incs.extend(_detect_cann_include_dirs(cann_root, bisheng_bin=resolved_bisheng))
    for inc in raw_incs:
        if (s := inc.strip()) and Path(s).is_dir() and (flag := f"-I{s}") not in seen:
            seen.add(flag)
            include_flags.append(flag)
    env = dict(os.environ)
    if (rc := _resolve_cann_root(cann_root, bisheng_bin=resolved_bisheng)) is not None and rc.is_dir() and not env.get("ASCEND_HOME_PATH"):
        env["ASCEND_HOME_PATH"] = str(rc).strip()
    return _AscendcCompileContext(extra_flags, "-x" in extra_flags, std_flags, include_flags, env)


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f".{dst.name}.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        tmp_path.unlink(missing_ok=True)


def compute_trackb_shared_so_filename(cpp_source: str, compile_flags: Sequence[str] | str | None = None) -> str:
    """Compute canonical cached shared-object filename for a Track-B TU and flags."""
    non_inc = [f for f in _normalize_flag_list(compile_flags) if not f.startswith("-I")]
    return f"trackb_{hashlib.sha256((cpp_source + '|' + ' '.join(non_inc)).encode('utf-8')).hexdigest()[:16]}.so"


def _build_trackb_shared_link_cmd(resolved_bisheng: str, *, arch: str, ctx: _AscendcCompileContext, src_path: Path, so_path: Path) -> list[str]:
    return [
        resolved_bisheng, "-O2", "-fPIC", *ctx.std_flags, "-mllvm", "-cce-aicore-dcpreload-args=false",
        f"--npu-arch={arch}", *ctx.include_flags, *ctx.extra_flags, "-Wl,--no-mmap-output-file", "-shared",
        str(src_path), "-o", str(so_path),
    ]


def _resolve_artifact_store(artifact_store: ArtifactStore | None) -> ArtifactStore:
    return artifact_store if artifact_store is not None else ArtifactStore(resolve_unified_tl_cache_dir())


def compile_trackb_shared_so(
    cpp_source: str, *, cache_dir: str | Path, compile_flags: Sequence[str] | str | None = None,
    arch: str = "dav-3510", bisheng_bin: str | Path | None = None, cann_root: str | Path | None = None,
    extra_include_dirs: Sequence[str | Path] | str | Path | None = None, timeout_sec: int = 120,
    subproc_run: Callable[..., SubprocResultLike] | None = None,
) -> tuple[bool, Path | None, str]:
    """Compile a Track-B C++20 TU into a content-addressed cached .so via bisheng."""
    if not isinstance(cpp_source, str) or not cpp_source.strip():
        return (False, None, "stage_b_ascendc_empty_source")
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    target_so = cache_root / compute_trackb_shared_so_filename(cpp_source, compile_flags)
    if is_verified_binary_file(target_so):
        return (True, target_so, "")
    if (resolved_bisheng := _resolve_bisheng_bin(bisheng_bin, cann_root=cann_root)) is None:
        return (False, None, "stage_b_ascendc_bisheng_not_found")
    ctx = _prepare_ascendc_compile_context(compile_flags=compile_flags, cann_root=cann_root, resolved_bisheng=resolved_bisheng, extra_include_dirs=extra_include_dirs)
    with tempfile.TemporaryDirectory(prefix="tf_trackb_shared_") as tmp_dir:
        src_path = Path(tmp_dir) / ("kernel.cpp" if ctx.has_explicit_x else "kernel.asc")
        tmp_so = Path(tmp_dir) / "kernel.so"
        src_path.write_text(cpp_source, encoding="utf-8")
        try:
            proc = (subproc_run or subprocess.run)(
                _build_trackb_shared_link_cmd(resolved_bisheng, arch=arch, ctx=ctx, src_path=src_path, so_path=tmp_so),
                text=True, capture_output=True, cwd=tmp_dir, env=ctx.env, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return (False, None, f"stage_b_ascendc_shared_link_timeout:{timeout_sec}s")
        raw = (proc.stderr or "") + "\n" + (proc.stdout or "")
        if proc.returncode != 0 or not is_verified_binary_file(tmp_so):
            return (False, None, f"stage_b_ascendc_shared_link_failed (rc={proc.returncode}): {_bounded_compile_diagnostic(raw)}")
        _atomic_copy_file(tmp_so, target_so)
        return (True, target_so, "")


def run_stage_b_ascendc_compile(
    cpp_source: str, *, arch: str = "dav-3510", compile_flags: Sequence[str] | str | None = None,
    bisheng_bin: str | Path | None = None, cann_root: str | Path | None = None,
    extra_include_dirs: Sequence[str | Path] | str | Path | None = None, timeout_sec: int = 120,
    enable_device_compile: bool = True, verify_shared_link: bool = False,
    emit_shared_so_cache_dir: str | Path | None = None, artifact_store: ArtifactStore | None = None,
    artifact_id: str = "", regime_tag: str = "", spec_key: str = "",
    subproc_run: Callable[..., SubprocResultLike] | None = None,
    disassembly_profile_fn: Callable[[str, str], dict[str, object] | None] | None = None,
    bisheng_resolver: Callable[..., str | None] | None = None,
    verify_instantiated_fn: Callable[..., tuple[bool, str]] | None = None,
    disassemble_fn: Callable[..., str] | None = None,
) -> StageBResult:
    """Verify dav-3510 C++20 translation unit compilation via bisheng in an isolated tempdir."""
    if not enable_device_compile:
        raise ValueError("enable_device_compile=False bypass has been removed")
    t0 = time.monotonic()
    if not isinstance(cpp_source, str) or not cpp_source.strip():
        return _stage_b_fail(t0, "stage_b_ascendc_empty_source", "STAGE_B_ASCENDC_EMPTY_SOURCE")
    resolve_bs = bisheng_resolver or _resolve_bisheng_bin
    if (resolved_bisheng := resolve_bs(bisheng_bin, cann_root=cann_root)) is None:
        return _stage_b_fail(t0, "stage_b_ascendc_bisheng_not_found", "BISHENG_NOT_FOUND", fault_plane="INFRA_TOOLCHAIN")
    run_fn = subproc_run or subprocess.run
    ctx = _prepare_ascendc_compile_context(compile_flags=compile_flags, cann_root=cann_root, resolved_bisheng=resolved_bisheng, extra_include_dirs=extra_include_dirs)
    extra_flags = ctx.extra_flags
    store = _resolve_artifact_store(artifact_store)
    spec_chk = CompileTaskSpec(artifact_id or "trackb", "_trackb_tu", (), (), tuple(extra_flags), unit_kind="cce_cpp20", unit_source=cpp_source)
    with tempfile.TemporaryDirectory(prefix="tf_stage_b_ascendc_") as tmp_dir:
        src_path = Path(tmp_dir) / ("kernel.cpp" if ctx.has_explicit_x else "kernel.asc")
        obj_path, so_path = Path(tmp_dir) / "kernel.o", Path(tmp_dir) / "kernel.so"
        src_path.write_text(cpp_source, encoding="utf-8")
        should_link = verify_shared_link and 'extern "C"' in cpp_source and "<<<" in cpp_source and not ctx.has_explicit_x and bool(ctx.env.get("ASCEND_HOME_PATH"))
        cached_so = (Path(emit_shared_so_cache_dir).expanduser().resolve() / compute_trackb_shared_so_filename(cpp_source, extra_flags)) if (should_link and emit_shared_so_cache_dir is not None) else None
        cached_hit = cached_so is not None and is_verified_binary_file(cached_so)

        def _invoke(legacy: bool) -> SubprocResultLike:
            af = ["-x", "cce", f"--cce-aicore-arch={arch}"] if legacy else [f"--npu-arch={arch}"]
            df = [] if any("Wdeprecated-declarations" in str(f) for f in extra_flags) else ["-Wno-deprecated-declarations"]
            return run_fn([resolved_bisheng, *af, "--cce-aicore-only", "--cce-res-usage", "-O2", *ctx.std_flags, *ctx.include_flags, *df, *extra_flags, "-c", str(src_path), "-o", str(obj_path)], text=True, capture_output=True, cwd=tmp_dir, env=ctx.env, timeout=timeout_sec)

        link_exec, link_fut = None, None
        try:
            if should_link and not cached_hit:
                link_exec = ThreadPoolExecutor(max_workers=2)
                link_fut = link_exec.submit(run_fn, _build_trackb_shared_link_cmd(resolved_bisheng, arch=arch, ctx=ctx, src_path=src_path, so_path=so_path), text=True, capture_output=True, cwd=tmp_dir, env=ctx.env, timeout=timeout_sec)
            try:
                proc = _invoke(False)
                if proc.returncode != 0:
                    le = ((proc.stderr or "") + "\n" + (proc.stdout or "")).lower()
                    if "--npu-arch" in le and any(t in le for t in ("unknown", "unrecognized", "unsupported", "invalid")):
                        proc = _invoke(True)
            except subprocess.TimeoutExpired:
                return _stage_b_fail(t0, f"stage_b_ascendc_compile_timeout:{timeout_sec}s", "STAGE_B_ASCENDC_COMPILE_TIMEOUT")
            comb = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0 or not obj_path.is_file():
                sp, vl, summary = _parse_cce_res_usage_with_summary(comb, asm_or_dump_text=cpp_source)
                raw_d = proc.stderr or proc.stdout or "bisheng failed without diagnostics"
                return _stage_b_fail(
                    t0, f"stage_b_ascendc_compile_failed (rc={proc.returncode}): {_bounded_compile_diagnostic(raw_d)}",
                    _classify_stage_b_overflow_fault_code(raw_d, summary, "STAGE_B_ASCENDC_COMPILE_FAILED"),
                    spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=(extract_microarch_profile_from_cce(cpp_source).to_dict() if cpp_source.strip() else None),
                    full_diag=raw_d, cce=cpp_source,
                )
            inst_ok, inst_err = (verify_instantiated_fn or _verify_object_has_instantiated_kernel)(obj_path, bisheng_bin=resolved_bisheng, subproc_run=run_fn)
            od_txt = (disassemble_fn or _disassemble_object)(obj_path, bisheng_bin=resolved_bisheng, subproc_run=run_fn)
            tel_out = _augment_bisheng_telemetry_text(comb, od_txt)
            dump_txt = f"{od_txt}\n{cpp_source}" if od_txt.strip() else cpp_source
            sp, vl, summary = _parse_cce_res_usage_with_summary(tel_out, asm_or_dump_text=dump_txt)
            ma_prof = disassembly_profile_fn(cpp_source, od_txt) if disassembly_profile_fn is not None else extract_microarch_profile_from_cce(cpp_source).to_dict()
            if not inst_ok:
                return _stage_b_fail(t0, inst_err, "STAGE_B_ASCENDC_UNINSTANTIATED_TEMPLATE", spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof, cce=cpp_source)
            if has_unreserved_vf_stack_overflow(spec_chk, summary.stack_bytes, tel_out):
                return _stage_b_fail(t0, f"stage_b_ascendc_vf_stack_overflow: stack_bytes={summary.stack_bytes}", "VF_STACK_OVERFLOW", spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof, full_diag=tel_out, cce=cpp_source)
            so_opt: Path | None = None
            if should_link:
                if cached_hit:
                    so_opt = cached_so
                elif link_fut is not None:
                    try:
                        lp = link_fut.result()
                    except subprocess.TimeoutExpired:
                        return _stage_b_fail(t0, f"stage_b_ascendc_shared_link_timeout:{timeout_sec}s", "STAGE_B_ASCENDC_SHARED_LINK_TIMEOUT", spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof)
                    lraw = (lp.stderr or "") + "\n" + (lp.stdout or "")
                    if lp.returncode != 0 or not so_path.is_file():
                        return _stage_b_fail(t0, f"stage_b_ascendc_shared_link_failed (rc={lp.returncode}): {_bounded_compile_diagnostic(lraw)}", _classify_stage_b_overflow_fault_code(lraw, summary, "STAGE_B_ASCENDC_SHARED_LINK_FAILED"), spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof, full_diag=lraw)
                    if cached_so is not None:
                        _atomic_copy_file(so_path, cached_so)
                    so_opt = so_path
        finally:
            if link_fut is not None:
                try:
                    link_fut.result()
                except Exception:
                    pass
            if link_exec is not None:
                link_exec.shutdown(wait=True)

        recorded_aid, recorded_bins = "", ()
        if built := [p for p in (obj_path, so_opt) if p is not None and p.is_file() and p.stat().st_size > 0]:
            closure_src = _normalize_trackb_unit_source(cpp_source, regime_tag=regime_tag)
            recorded_aid = artifact_id or compute_compile_task_key(normalized_kernel_unit_src=closure_src, cann_target=arch, pass_configs=None, compile_flags=extra_flags, static_spec_signature=spec_key)
            rec = store.record_verified_artifact(recorded_aid, binary_paths=built, backend_kind="ascendc", arch=arch, flags=" ".join(extra_flags), spec_key=spec_key, closure_src=closure_src, metadata={"stack_bytes": int(summary.stack_bytes)}, copy_binaries=True)
            recorded_bins = rec.binary_paths
    return StageBResult(
        passed=True, elapsed_sec=time.monotonic() - t0, spill_bytes=sp, vloopv2_hit=vl,
        compiler_summary=summary, microarch_profile=ma_prof, generated_cce=cpp_source,
        artifact_id=recorded_aid, binary_paths=recorded_bins,
    )

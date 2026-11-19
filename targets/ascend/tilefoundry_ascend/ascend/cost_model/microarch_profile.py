"""Unified Track-A (TIR) and Track-B (CCE/AscendC) Ascend 950 microarchitectural profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
import re
from typing import Any


def _safe_int(val: object, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return default


def _safe_float(val: object, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default


@dataclass(frozen=True)
class MicroarchProfile:
    """Compiler-invariant hardware operation, DMA edge, and bank geometry telemetry."""

    # L1 AVE instruction & load/store balance counters (per representative kernel launch)
    fp_alu_ops: int = 0
    int_alu_ops: int = 0
    sfu_f16_ops: int = 0
    sfu_f32_ops: int = 0
    reduce_ops: int = 0
    slot0_only_ops: int = 0
    ub_load_bytes: int = 0
    ub_store_bytes: int = 0

    # Explicit 6-pipe directional storage-transition byte volumes (DAG2)
    gm_to_l1_bytes: int = 0
    gm_to_ub_bytes: int = 0
    l1_to_l0a_bytes: int = 0
    l1_to_l0b_bytes: int = 0
    l0c_to_ub_bytes: int = 0
    l0c_to_l1_bytes: int = 0
    l0c_to_gm_bytes: int = 0
    ub_to_l1_bytes: int = 0
    ub_to_gm_bytes: int = 0

    # Cube, sync & pipeline loop structure
    cube_mmad_count: int = 0
    cube_macs_total: int = 0
    cv_sync_barriers: int = 0
    pipe_all_barriers: int = 0
    pipeline_depth: int = 1
    inner_loop_trips: int = 1
    subkernel_count: int = 1

    # Memory bank conflict multipliers (>= 1.0)
    ub_bank_conflict_factor: float = 1.0
    l1_bank_conflict_factor: float = 1.0
    simt_shm_conflict_factor: float = 1.0

    source_kind: str = "empty"

    @property
    def is_empty(self) -> bool:
        return self.source_kind == "empty" and (
            self.total_transfer_bytes == 0
            and self.total_vector_ops == 0
            and self.cube_mmad_count == 0
            and self.cube_macs_total == 0
        )

    @property
    def total_vector_ops(self) -> int:
        return (
            self.fp_alu_ops
            + self.int_alu_ops
            + self.sfu_f16_ops
            + self.sfu_f32_ops
            + self.reduce_ops
            + self.slot0_only_ops
        )

    @property
    def total_transfer_bytes(self) -> int:
        return (
            self.gm_to_l1_bytes
            + self.gm_to_ub_bytes
            + self.l1_to_l0a_bytes
            + self.l1_to_l0b_bytes
            + self.l0c_to_ub_bytes
            + self.l0c_to_l1_bytes
            + self.l0c_to_gm_bytes
            + self.ub_to_l1_bytes
            + self.ub_to_gm_bytes
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize profile to a JSON-safe dictionary."""
        return asdict(self)

    @classmethod
    def coerce(cls, raw: MicroarchProfile | Mapping[str, object] | None) -> MicroarchProfile:
        """Coerce Mapping or MicroarchProfile (returning empty profile for None)."""
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(f"Expected MicroarchProfile or Mapping, got {type(raw).__name__}")
        valid = set(cls.__dataclass_fields__.keys())
        kw: dict[str, Any] = {}
        for k in valid:
            if k not in raw or raw[k] is None:
                continue
            if k == "source_kind":
                kw[k] = str(raw[k])
            elif k.endswith("_factor"):
                kw[k] = max(1.0, _safe_float(raw[k], 1.0))
            elif k in ("pipeline_depth", "inner_loop_trips", "subkernel_count"):
                kw[k] = max(1, _safe_int(raw[k], 1))
            else:
                kw[k] = max(0, _safe_int(raw[k], 0))
        return cls(**kw)


def _eval_tirx_expr_int(expr: object, var_bindings: Mapping[str, int] | None) -> int | None:
    if expr is None:
        return None
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return int(expr)
    if hasattr(expr, "value") and not callable(getattr(expr, "value")):
        try:
            return int(expr.value)
        except Exception:
            pass
    if var_bindings:
        name = getattr(expr, "name_hint", None) or getattr(expr, "name", None)
        if isinstance(name, str) and name in var_bindings:
            return int(var_bindings[name])
    return None


def _map_tl_scope_name(raw_scope: object) -> str | None:
    s = str(raw_scope or "").strip().lower()
    if s in ("global", "global.ddr"):
        return "GM"
    if s in ("shared.l1", "l1"):
        return "L1"
    if s in ("shared.l0a", "l0a"):
        return "L0A"
    if s in ("shared.l0b", "l0b"):
        return "L0B"
    if s in ("shared.l0c", "l0c"):
        return "L0C"
    if s in ("shared.ss", "ssbuf"):
        return "SSBUF"
    if s in ("shared", "shared.dyn", "ub"):
        return "UB"
    return None


def _extract_region_info(
    arg: object,
    *,
    tirx: Any,
    DataTypeCls: Any,
    var_bindings: Mapping[str, int] | None,
) -> tuple[str | None, int, int, str]:
    """Return (scope, total_bytes, elems, dtype_str) from a BufferLoad / region / access_ptr."""
    if isinstance(arg, tirx.Call):
        op_name = str(getattr(arg.op, "name", arg.op)).lower()
        args = list(arg.args)
        if "region" in op_name and args:
            bl = args[0]
            buf = getattr(bl, "buffer", None)
            extents = args[2:]
            if buf is not None:
                scope = _map_tl_scope_name(buf.scope())
                dtype_str = str(getattr(buf, "dtype", "float16"))
                try:
                    elem_bits = int(DataTypeCls(dtype_str).bits)
                except Exception:
                    elem_bits = 16
                elems = 1
                known = True
                for ext in extents:
                    iv = _eval_tirx_expr_int(ext, var_bindings)
                    if iv is None or iv <= 0:
                        known = False
                        break
                    elems *= iv
                if not known or not extents:
                    elems = 1
                    for ext in getattr(buf, "shape", ()) or ():
                        iv = _eval_tirx_expr_int(ext, var_bindings)
                        if iv is None or iv <= 0:
                            elems = 0
                            break
                        elems *= iv
                total_b = (elems * elem_bits + 7) // 8 if elems > 0 else 0
                return scope, total_b, max(0, elems), dtype_str
        if "access_ptr" in op_name and args:
            bl = args[0]
            buf = getattr(bl, "buffer", None)
            if buf is not None:
                scope = _map_tl_scope_name(buf.scope())
                dtype_str = str(getattr(buf, "dtype", "float16"))
                try:
                    elem_bits = int(DataTypeCls(dtype_str).bits)
                except Exception:
                    elem_bits = 16
                iv = _eval_tirx_expr_int(args[3], var_bindings) if len(args) > 3 else None
                if iv is None or iv <= 0:
                    elems = 1
                    for ext in getattr(buf, "shape", ()) or ():
                        d = _eval_tirx_expr_int(ext, var_bindings)
                        if d is None or d <= 0:
                            elems = 0
                            break
                        elems *= d
                else:
                    elems = iv
                total_b = (elems * elem_bits + 7) // 8 if elems > 0 else 0
                return scope, total_b, max(0, elems), dtype_str
    buf = getattr(arg, "buffer", None)
    if buf is not None:
        scope = _map_tl_scope_name(buf.scope())
        dtype_str = str(getattr(buf, "dtype", "float16"))
        try:
            elem_bits = int(DataTypeCls(dtype_str).bits)
        except Exception:
            elem_bits = 16
        elems = 1
        for ext in getattr(buf, "shape", ()) or ():
            d = _eval_tirx_expr_int(ext, var_bindings)
            if d is None or d <= 0:
                elems = 0
                break
            elems *= d
        total_b = (elems * elem_bits + 7) // 8 if elems > 0 else 0
        return scope, total_b, max(0, elems), dtype_str
    return None, 0, 0, ""


def extract_microarch_profile_from_tirx(
    prim_func: object,
    *,
    var_bindings: Mapping[str, int] | None = None,
) -> MicroarchProfile:
    """Extract Ascend 950 MicroarchProfile from a TileLang PrimFunc or IRModule."""
    if prim_func is None:
        return MicroarchProfile()
    try:
        from tilelang import tvm
        from tilelang.utils.language import DataType as TLDataType
        from tvm import tirx
    except Exception:
        return MicroarchProfile()

    funcs: list[Any] = []
    if isinstance(prim_func, tvm.IRModule):
        funcs = [fn for fn in prim_func.functions.values() if isinstance(fn, tirx.PrimFunc)]
    elif isinstance(prim_func, tirx.PrimFunc):
        funcs = [prim_func]
    if not funcs:
        return MicroarchProfile()

    c = {
        "fp_alu_ops": 0,
        "int_alu_ops": 0,
        "sfu_f16_ops": 0,
        "sfu_f32_ops": 0,
        "reduce_ops": 0,
        "slot0_only_ops": 0,
        "ub_load_bytes": 0,
        "ub_store_bytes": 0,
        "gm_to_l1_bytes": 0,
        "gm_to_ub_bytes": 0,
        "l1_to_l0a_bytes": 0,
        "l1_to_l0b_bytes": 0,
        "l0c_to_ub_bytes": 0,
        "l0c_to_l1_bytes": 0,
        "l0c_to_gm_bytes": 0,
        "ub_to_l1_bytes": 0,
        "ub_to_gm_bytes": 0,
        "cube_mmad_count": 0,
        "cube_macs_total": 0,
        "cv_sync_barriers": 0,
        "pipe_all_barriers": 0,
        "pipeline_depth": 1,
        "inner_loop_trips": 1,
        "subkernel_count": len(funcs),
    }
    ub_total_bytes = 0
    ub_conflict_bytes = 0
    l1_buffers: list[int] = []

    def _classify_dtype_family(dtype_str: str) -> str:
        low = str(dtype_str or "").lower()
        if "float16" in low or "bfloat16" in low or low.startswith(("f16", "bf16")):
            return "f16"
        if "int" in low or "uint" in low or low == "bool":
            return "int"
        return "f32"

    def _record_buf(buf: object) -> None:
        nonlocal ub_total_bytes, ub_conflict_bytes
        scope = _map_tl_scope_name(buf.scope()) if hasattr(buf, "scope") else None
        if scope is None:
            return
        shape = getattr(buf, "shape", ()) or ()
        dims: list[int] = []
        for ext in shape:
            iv = _eval_tirx_expr_int(ext, var_bindings)
            if iv is None or iv <= 0:
                dims = []
                break
            dims.append(iv)
        if not dims:
            return
        try:
            bits = int(TLDataType(str(buf.dtype)).bits)
        except Exception:
            bits = 16
        total_b = (math.prod(dims) * bits + 7) // 8
        inner_b = (dims[-1] * bits + 7) // 8
        if scope == "UB" and len(dims) >= 2 and math.prod(dims[:-1]) > 1:
            ub_total_bytes += total_b
            if inner_b >= 512 and inner_b % 512 == 0:
                ub_conflict_bytes += total_b
        elif scope == "L1":
            l1_buffers.append(total_b)

    def _scan_expr(expr: object, *, repeat: int) -> str:
        if expr is None:
            return ""
        dt_str = str(getattr(expr, "dtype", ""))
        if isinstance(expr, tirx.Call):
            op_name = str(getattr(expr.op, "name", expr.op)).lower()
            args = list(expr.args)
            arg_dtypes = [_scan_expr(a, repeat=repeat) for a in args]
            eff_dt = next((d for d in arg_dtypes if d), dt_str)
            fam = _classify_dtype_family(eff_dt)
            if any(tok in op_name for tok in ("vintlv", "vdintlv", "interleave")):
                c["slot0_only_ops"] += repeat
            elif any(tok in op_name for tok in ("exp", "log", "sqrt", "rsqrt", "div", "tanh", "erf", "sigmoid")):
                if fam == "f16":
                    c["sfu_f16_ops"] += repeat
                else:
                    c["sfu_f32_ops"] += repeat
            elif "reduce" in op_name:
                c["reduce_ops"] += repeat
            elif "simd" in op_name or "vec" in op_name or "cast" in op_name or "fma" in op_name:
                if fam == "int":
                    c["int_alu_ops"] += repeat
                else:
                    c["fp_alu_ops"] += repeat
            return eff_dt
        if isinstance(expr, tirx.BufferLoad):
            scope = _map_tl_scope_name(expr.buffer.scope())
            if scope == "UB":
                try:
                    b = max(2, int(TLDataType(str(expr.dtype)).bits) * int(TLDataType(str(expr.dtype)).lanes) // 8)
                except Exception:
                    b = 2
                c["ub_load_bytes"] += repeat * b
            return str(getattr(expr, "dtype", ""))
        if isinstance(expr, (tirx.Add, tirx.Sub, tirx.Mul, tirx.Div, tirx.FloorDiv, tirx.Mod, tirx.Min, tirx.Max)):
            ld = _scan_expr(expr.a, repeat=repeat)
            rd = _scan_expr(expr.b, repeat=repeat)
            fam = _classify_dtype_family(ld or rd or dt_str)
            if fam == "int":
                c["int_alu_ops"] += repeat
            else:
                c["fp_alu_ops"] += repeat
            return ld or rd or dt_str
        return dt_str

    def _walk(stmt: object, *, repeat: int = 1) -> None:
        if stmt is None:
            return
        if isinstance(stmt, tirx.SeqStmt):
            for ch in stmt.seq:
                _walk(ch, repeat=repeat)
            return
        if isinstance(stmt, tirx.For):
            ann = getattr(stmt, "annotations", None) or {}
            raw_stages = ann.get("num_stages", ann.get("software_pipeline_stage"))
            if raw_stages is not None:
                c["pipeline_depth"] = max(c["pipeline_depth"], max(1, _safe_int(raw_stages, 1)))
            tb = getattr(stmt, "thread_binding", None)
            if tb is not None:
                _walk(stmt.body, repeat=repeat)
                return
            trip = _eval_tirx_expr_int(stmt.extent, var_bindings)
            eff_trip = trip if (trip is not None and trip > 0) else 1
            if eff_trip > 1:
                c["inner_loop_trips"] = max(c["inner_loop_trips"], eff_trip)
            _walk(stmt.body, repeat=repeat * eff_trip)
            return
        if isinstance(stmt, tirx.While):
            _walk(stmt.body, repeat=repeat)
            return
        if isinstance(stmt, tirx.IfThenElse):
            _walk(stmt.then_case, repeat=repeat)
            if stmt.else_case is not None:
                _walk(stmt.else_case, repeat=repeat)
            return
        if isinstance(stmt, tirx.AttrStmt):
            ann_val = _safe_int(getattr(stmt, "value", None), 0)
            if "stage" in str(getattr(stmt, "attr_key", "")).lower() and ann_val > 1:
                c["pipeline_depth"] = max(c["pipeline_depth"], ann_val)
            _walk(stmt.body, repeat=repeat)
            return
        if isinstance(stmt, (tirx.SBlockRealize,)):
            _walk(stmt.block, repeat=repeat)
            return
        if isinstance(stmt, tirx.SBlock):
            for buf in getattr(stmt, "alloc_buffers", ()) or ():
                _record_buf(buf)
            _walk(stmt.body, repeat=repeat)
            return
        if isinstance(stmt, (tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(stmt, "buffer"):
                _record_buf(stmt.buffer)
            if hasattr(stmt, "body"):
                _walk(stmt.body, repeat=repeat)
            return
        if isinstance(stmt, tirx.Bind):
            _scan_expr(stmt.value, repeat=repeat)
            if hasattr(stmt, "body"):
                _walk(stmt.body, repeat=repeat)
            return
        if isinstance(stmt, tirx.BufferStore):
            scope = _map_tl_scope_name(stmt.buffer.scope())
            if scope == "UB":
                try:
                    dt = TLDataType(str(stmt.value.dtype))
                    b = max(2, (int(dt.bits) * int(dt.lanes) + 7) // 8)
                except Exception:
                    b = 2
                c["ub_store_bytes"] += repeat * b
            _scan_expr(stmt.value, repeat=repeat)
            return
        if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            op_name = str(getattr(stmt.value.op, "name", stmt.value.op)).lower()
            args = list(stmt.value.args)
            if "copy" in op_name and len(args) >= 2:
                src_s, src_b, _, _ = _extract_region_info(
                    args[0], tirx=tirx, DataTypeCls=TLDataType, var_bindings=var_bindings
                )
                dst_s, dst_b, _, _ = _extract_region_info(
                    args[1], tirx=tirx, DataTypeCls=TLDataType, var_bindings=var_bindings
                )
                xfer = max(src_b, dst_b)
                edge_key = {
                    ("GM", "L1"): "gm_to_l1_bytes",
                    ("GM", "UB"): "gm_to_ub_bytes",
                    ("L1", "L0A"): "l1_to_l0a_bytes",
                    ("L1", "L0B"): "l1_to_l0b_bytes",
                    ("L0C", "UB"): "l0c_to_ub_bytes",
                    ("L0C", "L1"): "l0c_to_l1_bytes",
                    ("L0C", "GM"): "l0c_to_gm_bytes",
                    ("UB", "L1"): "ub_to_l1_bytes",
                    ("UB", "GM"): "ub_to_gm_bytes",
                }.get((src_s, dst_s))
                if edge_key is not None and xfer > 0:
                    c[edge_key] += repeat * xfer
                if src_s == "UB":
                    c["ub_load_bytes"] += repeat * xfer
                if dst_s == "UB":
                    c["ub_store_bytes"] += repeat * xfer
                return
            if ("gemm" in op_name or "mmad" in op_name) and len(args) >= 2:
                c["cube_mmad_count"] += repeat
                _s0, _b0, elems_a, _dt0 = _extract_region_info(
                    args[0], tirx=tirx, DataTypeCls=TLDataType, var_bindings=var_bindings
                )
                _s1, _b1, elems_b, _dt1 = _extract_region_info(
                    args[1], tirx=tirx, DataTypeCls=TLDataType, var_bindings=var_bindings
                )
                _s2, _b2, elems_c, _dt2 = (
                    _extract_region_info(
                        args[2], tirx=tirx, DataTypeCls=TLDataType, var_bindings=var_bindings
                    )
                    if len(args) >= 3
                    else (None, 0, 0, "")
                )
                # Try inferring exact M*N*K when buffer shapes are known; else approximate safely.
                m_val, n_val, k_val = 0, 0, 0
                buf_a = getattr(getattr(args[0], "args", [None])[0] if isinstance(args[0], tirx.Call) else args[0], "buffer", None)
                buf_b = getattr(getattr(args[1], "args", [None])[0] if isinstance(args[1], tirx.Call) else args[1], "buffer", None)
                if buf_a is not None and buf_b is not None:
                    sa = [_eval_tirx_expr_int(d, var_bindings) for d in (getattr(buf_a, "shape", ()) or ())]
                    sb = [_eval_tirx_expr_int(d, var_bindings) for d in (getattr(buf_b, "shape", ()) or ())]
                    if len(sa) >= 2 and len(sb) >= 2 and sa[-2] and sa[-1] and sb[-2] and sb[-1]:
                        m_val = sa[-2]
                        k_val = sa[-1]
                        n_val = sb[-1] if sb[-2] == k_val else sb[-2]
                macs = (m_val * n_val * k_val) if (m_val > 0 and n_val > 0 and k_val > 0) else max(elems_a, elems_b, elems_c, 0) * 64
                c["cube_macs_total"] += repeat * macs
                return
            if "cross_flag" in op_name or "cross_core" in op_name:
                c["cv_sync_barriers"] += repeat
                return
            if "pipe_barrier" in op_name or "pipe_all" in op_name:
                c["pipe_all_barriers"] += repeat
                return
            _scan_expr(stmt.value, repeat=repeat)

    for fn in funcs:
        for buf in (getattr(fn, "buffer_map", {}) or {}).values():
            _record_buf(buf)
        _walk(fn.body, repeat=1)

    ub_factor = (
        1.0 + 0.5 * (ub_conflict_bytes / float(ub_total_bytes))
        if ub_total_bytes > 0 and ub_conflict_bytes > 0
        else 1.0
    )
    l1_sum = sum(l1_buffers)
    l1_half = 256 * 1024
    l1_factor = 1.25 if l1_sum > l1_half else (1.12 if len(l1_buffers) >= 2 and l1_sum > l1_half // 2 else 1.0)

    return MicroarchProfile(
        **c,
        ub_bank_conflict_factor=round(ub_factor, 4),
        l1_bank_conflict_factor=round(l1_factor, 4),
        simt_shm_conflict_factor=1.0,
        source_kind="tirx",
    )


def _strip_cpp_comments(text: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", no_block)


def extract_microarch_profile_from_cce(source_text: str) -> MicroarchProfile:
    """Extract unified post-codegen MicroarchProfile from Track A lowered CCE or Track B AscendC."""
    raw = str(source_text or "")
    if not raw.strip():
        return MicroarchProfile()
    clean = _strip_cpp_comments(raw)

    subkernels = len(re.findall(r"\b__aicore__\b", clean))
    cube_mmad = len(re.findall(r"\b(?:Mmad|mmad|mad|cube::gemm|gemm_v\d)\b\s*[(<]", clean))
    cv_sync = len(re.findall(r"\bCrossCore(?:Set|Wait)Flag\b", clean))
    pipe_all = len(re.findall(r"\bpipe_barrier\s*\(\s*PIPE_ALL\s*\)", clean))
    reduce_cnt = len(re.findall(r"\b(?:Reduce(?:Max|Min|Sum)|vcmax|vcmin|vcadd|reduce_(?:sum|max))\b", clean, flags=re.IGNORECASE))
    slot0_cnt = len(re.findall(r"\b(?:vintlv|vdintlv|Interleave|DeInterleave)\b", clean, flags=re.IGNORECASE))

    sfu_matches = re.findall(
        r"([^\n;]{0,80}\b(?:vexp|vln|vsqrt|vrsqrt|vdiv|Exp|Ln|Sqrt|Rsqrt|Div|Tanh|Sigmoid)\b[^\n;]{0,80})",
        clean,
        flags=re.IGNORECASE,
    )
    sfu_f16 = 0
    sfu_f32 = 0
    for snippet in sfu_matches:
        low = snippet.lower()
        if any(tok in low for tok in ("half", "bfloat16", "fp16", "bf16", "__fp16")):
            sfu_f16 += 1
        else:
            sfu_f32 += 1

    fp_alu = len(re.findall(r"\b(?:vadd|vmul|vsub|vmad|vfma|vfadd|vfmul|Add|Mul|Sub|Muls|Adds)\b", clean))
    int_alu = len(re.findall(r"\b(?:viadd|vimul|vand|vor|vshl|vshr)\b", clean))

    # Transfer-op occurrence counts scaled by typical single-tile packet (32 KiB) only when no TIR volume exists.
    unit_xfer = 32768
    gm_l1 = len(re.findall(r"\bcopy_gm_to_cbuf\b", clean)) * unit_xfer
    gm_ub = len(re.findall(r"\bcopy_gm_to_ubuf\b", clean)) * unit_xfer
    l1_l0a = len(re.findall(r"\bcopy_cbuf_to_ca\b", clean)) * unit_xfer
    l1_l0b = len(re.findall(r"\bcopy_cbuf_to_cb\b", clean)) * unit_xfer
    l0c_ub = len(re.findall(r"\bcopy_matrix_cc_to_ubuf\b", clean)) * unit_xfer
    l0c_l1 = len(re.findall(r"\bcopy_matrix_cc_to_cbuf\b", clean)) * unit_xfer
    l0c_gm = len(re.findall(r"\bcopy_matrix_cc_to_gm\b|\bFixpipe\b", clean)) * unit_xfer
    ub_l1 = len(re.findall(r"\bcopy_ubuf_to_cbuf\b", clean)) * unit_xfer
    ub_gm = len(re.findall(r"\bcopy_ubuf_to_gm\b", clean)) * unit_xfer

    # Detect SIMT shared-memory power-of-2 innermost stride patterns if present.
    simt_factor = 1.0
    if re.search(r"__simt_vf__|__shared__", clean):
        shm_strs = re.findall(r"__shared__[^;]*\[(\d+)\]\s*\[(\d+)\]", clean)
        for _rows, cols in shm_strs:
            if int(cols) in (16, 32, 64, 128):
                simt_factor = max(simt_factor, 1.5)

    return MicroarchProfile(
        fp_alu_ops=fp_alu,
        int_alu_ops=int_alu,
        sfu_f16_ops=sfu_f16,
        sfu_f32_ops=sfu_f32,
        reduce_ops=reduce_cnt,
        slot0_only_ops=slot0_cnt,
        ub_load_bytes=ub_l1 + ub_gm,
        ub_store_bytes=gm_ub + l0c_ub,
        gm_to_l1_bytes=gm_l1,
        gm_to_ub_bytes=gm_ub,
        l1_to_l0a_bytes=l1_l0a,
        l1_to_l0b_bytes=l1_l0b,
        l0c_to_ub_bytes=l0c_ub,
        l0c_to_l1_bytes=l0c_l1,
        l0c_to_gm_bytes=l0c_gm,
        ub_to_l1_bytes=ub_l1,
        ub_to_gm_bytes=ub_gm,
        cube_mmad_count=cube_mmad,
        cube_macs_total=0,
        cv_sync_barriers=cv_sync,
        pipe_all_barriers=pipe_all,
        pipeline_depth=1,
        inner_loop_trips=1,
        subkernel_count=max(1, subkernels),
        ub_bank_conflict_factor=1.0,
        l1_bank_conflict_factor=1.0,
        simt_shm_conflict_factor=simt_factor,
        source_kind="cce",
    )


def merge_microarch_profiles(
    stage_a: MicroarchProfile | Mapping[str, object] | None,
    stage_b: MicroarchProfile | Mapping[str, object] | None,
) -> MicroarchProfile:
    """Merge Stage-A TIR profile (exact byte volumes) with Stage-B CCE/AscendC profile."""
    pa = MicroarchProfile.coerce(stage_a)
    pb = MicroarchProfile.coerce(stage_b)
    if pa.is_empty and pb.is_empty:
        return MicroarchProfile()
    if pa.is_empty:
        return pb
    if pb.is_empty:
        return pa

    def _prefer_tir_bytes(a_val: int, b_val: int) -> int:
        return a_val if a_val > 0 else b_val

    return MicroarchProfile(
        fp_alu_ops=max(pa.fp_alu_ops, pb.fp_alu_ops),
        int_alu_ops=max(pa.int_alu_ops, pb.int_alu_ops),
        sfu_f16_ops=max(pa.sfu_f16_ops, pb.sfu_f16_ops),
        sfu_f32_ops=max(pa.sfu_f32_ops, pb.sfu_f32_ops),
        reduce_ops=max(pa.reduce_ops, pb.reduce_ops),
        slot0_only_ops=max(pa.slot0_only_ops, pb.slot0_only_ops),
        ub_load_bytes=_prefer_tir_bytes(pa.ub_load_bytes, pb.ub_load_bytes),
        ub_store_bytes=_prefer_tir_bytes(pa.ub_store_bytes, pb.ub_store_bytes),
        gm_to_l1_bytes=_prefer_tir_bytes(pa.gm_to_l1_bytes, pb.gm_to_l1_bytes),
        gm_to_ub_bytes=_prefer_tir_bytes(pa.gm_to_ub_bytes, pb.gm_to_ub_bytes),
        l1_to_l0a_bytes=_prefer_tir_bytes(pa.l1_to_l0a_bytes, pb.l1_to_l0a_bytes),
        l1_to_l0b_bytes=_prefer_tir_bytes(pa.l1_to_l0b_bytes, pb.l1_to_l0b_bytes),
        l0c_to_ub_bytes=_prefer_tir_bytes(pa.l0c_to_ub_bytes, pb.l0c_to_ub_bytes),
        l0c_to_l1_bytes=_prefer_tir_bytes(pa.l0c_to_l1_bytes, pb.l0c_to_l1_bytes),
        l0c_to_gm_bytes=_prefer_tir_bytes(pa.l0c_to_gm_bytes, pb.l0c_to_gm_bytes),
        ub_to_l1_bytes=_prefer_tir_bytes(pa.ub_to_l1_bytes, pb.ub_to_l1_bytes),
        ub_to_gm_bytes=_prefer_tir_bytes(pa.ub_to_gm_bytes, pb.ub_to_gm_bytes),
        cube_mmad_count=max(pa.cube_mmad_count, pb.cube_mmad_count),
        cube_macs_total=max(pa.cube_macs_total, pb.cube_macs_total),
        cv_sync_barriers=max(pa.cv_sync_barriers, pb.cv_sync_barriers),
        pipe_all_barriers=max(pa.pipe_all_barriers, pb.pipe_all_barriers),
        pipeline_depth=max(pa.pipeline_depth, pb.pipeline_depth),
        inner_loop_trips=max(pa.inner_loop_trips, pb.inner_loop_trips),
        subkernel_count=max(pa.subkernel_count, pb.subkernel_count),
        ub_bank_conflict_factor=max(pa.ub_bank_conflict_factor, pb.ub_bank_conflict_factor),
        l1_bank_conflict_factor=max(pa.l1_bank_conflict_factor, pb.l1_bank_conflict_factor),
        simt_shm_conflict_factor=max(pa.simt_shm_conflict_factor, pb.simt_shm_conflict_factor),
        source_kind="merged",
    )


__all__ = [
    "MicroarchProfile",
    "extract_microarch_profile_from_cce",
    "extract_microarch_profile_from_tirx",
    "merge_microarch_profiles",
]

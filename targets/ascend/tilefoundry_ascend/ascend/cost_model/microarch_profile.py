"""Unified Track-A (TIR) and Track-B (CCE/AscendC) Ascend 950 microarchitectural profile."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from tilefoundry_ascend.ascend.json_record import JsonRecordMixin

_KNOWN_ASCEND_DTYPE_BYTES: dict[str, float] = {
    "f16": 2.0, "fp16": 2.0, "float16": 2.0, "bf16": 2.0, "bfloat16": 2.0,
    "f32": 4.0, "fp32": 4.0, "float32": 4.0, "hf32": 4.0, "f64": 8.0, "fp64": 8.0, "float64": 8.0,
    "fp8e4m3": 1.0, "fp8_e4m3": 1.0, "float8_e4m3": 1.0, "float8_e4m3fn": 1.0, "e4m3": 1.0,
    "fp8e5m2": 1.0, "fp8_e5m2": 1.0, "float8_e5m2": 1.0, "e5m2": 1.0, "hif8": 1.0,
    "i8": 1.0, "int8": 1.0, "u8": 1.0, "uint8": 1.0, "i16": 2.0, "int16": 2.0, "u16": 2.0, "uint16": 2.0,
    "i32": 4.0, "int32": 4.0, "u32": 4.0, "uint32": 4.0, "i64": 8.0, "int64": 8.0, "u64": 8.0, "uint64": 8.0,
    "f4e2m1": 0.5, "f4e1m2": 0.5, "fp4": 0.5, "i4": 0.5, "int4": 0.5, "u4": 0.5, "uint4": 0.5,
    "bool": 1.0, "int1": 1.0,
}
_TL_SCOPE_MAP: dict[str, str] = {
    "global": "GM", "global.ddr": "GM", "shared.l1": "L1", "l1": "L1", "shared.l0a": "L0A", "l0a": "L0A",
    "shared.l0b": "L0B", "l0b": "L0B", "shared.l0c": "L0C", "l0c": "L0C", "shared.ss": "SSBUF",
    "ssbuf": "SSBUF", "shared": "UB", "shared.dyn": "UB", "ub": "UB",
}
_COPY_EDGE_KEYS: dict[tuple[str, str], str] = {
    ("GM", "L1"): "gm_to_l1_bytes", ("GM", "UB"): "gm_to_ub_bytes",
    ("L1", "L0A"): "l1_to_l0a_bytes", ("L1", "L0B"): "l1_to_l0b_bytes",
    ("L0C", "UB"): "l0c_to_ub_bytes", ("L0C", "L1"): "l0c_to_l1_bytes",
    ("L0C", "GM"): "l0c_to_gm_bytes", ("UB", "L1"): "ub_to_l1_bytes", ("UB", "GM"): "ub_to_gm_bytes",
}
_BYTE_FIELDS = frozenset(_COPY_EDGE_KEYS.values()) | {"ub_load_bytes", "ub_store_bytes"}
_MIN1_FIELDS = frozenset({"pipeline_depth", "inner_loop_trips", "subkernel_count"})
_CCE_XFER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gm_to_l1_bytes", r"\b(?:copy_gm_to_cbuf|asc_copy_gm2l1\w*)\b"),
    ("gm_to_ub_bytes", r"\b(?:copy_gm_to_ubuf|asc_copy_gm2ub\w*)\b"),
    ("l1_to_l0a_bytes", r"\bcopy_cbuf_to_ca\b"), ("l1_to_l0b_bytes", r"\bcopy_cbuf_to_cb\b"),
    ("l0c_to_ub_bytes", r"\b(?:copy_matrix_cc_to_ubuf|asc_copy_l0c2ub\w*)\b"),
    ("l0c_to_l1_bytes", r"\bcopy_matrix_cc_to_cbuf\b"),
    ("l0c_to_gm_bytes", r"\b(?:copy_matrix_cc_to_gm|Fixpipe|asc_copy_l0c2gm\w*)\b"),
    ("ub_to_l1_bytes", r"\b(?:copy_ubuf_to_cbuf|asc_copy_ub2l1\w*)\b"),
    ("ub_to_gm_bytes", r"\b(?:copy_ubuf_to_gm|asc_copy_ub2gm\w*)\b"),
)


def _dtype_bytes_strict(dtype_name: object) -> float:
    raw = str(getattr(dtype_name, "name", dtype_name) or "").strip().lower()
    if raw in _KNOWN_ASCEND_DTYPE_BYTES:
        return _KNOWN_ASCEND_DTYPE_BYTES[raw]
    m = re.fullmatch(r"([a-z0-9_]+)x(\d+)", raw)
    if m and m.group(1) in _KNOWN_ASCEND_DTYPE_BYTES and int(m.group(2)) > 0:
        return _KNOWN_ASCEND_DTYPE_BYTES[m.group(1)] * int(m.group(2))
    raise ValueError(f"Unknown Ascend dtype {dtype_name!r}; refusing 16-bit silent fallback")


def _dtype_bits(dtype_name: object, dt_cls: Any = None) -> int:
    if dt_cls is not None:
        try:
            return int(dt_cls(str(dtype_name)).bits)
        except Exception:
            pass
    return max(1, int(round(_dtype_bytes_strict(dtype_name) * 8.0)))


def _safe_num(val: object, default: float | int, is_float: bool = False) -> Any:
    try:
        v = float(val) if is_float else int(val)  # type: ignore[arg-type]
        return (v if math.isfinite(v) else default) if is_float else v
    except Exception:
        return default


@dataclass(frozen=True)
class MicroarchProfile(JsonRecordMixin):
    """Compiler-invariant hardware operation, DMA edge, and bank geometry telemetry."""

    fp_alu_ops: int = 0
    int_alu_ops: int = 0
    sfu_f16_ops: int = 0
    sfu_f32_ops: int = 0
    reduce_ops: int = 0
    slot0_only_ops: int = 0
    scalar_ops: int = 0
    ub_load_bytes: int = 0
    ub_store_bytes: int = 0
    gm_to_l1_bytes: int = 0
    gm_to_ub_bytes: int = 0
    l1_to_l0a_bytes: int = 0
    l1_to_l0b_bytes: int = 0
    l0c_to_ub_bytes: int = 0
    l0c_to_l1_bytes: int = 0
    l0c_to_gm_bytes: int = 0
    ub_to_l1_bytes: int = 0
    ub_to_gm_bytes: int = 0
    cube_mmad_count: int = 0
    cube_macs_total: int = 0
    cv_sync_barriers: int = 0
    pipe_all_barriers: int = 0
    pipeline_depth: int = 1
    inner_loop_trips: int = 1
    subkernel_count: int = 1
    ub_bank_conflict_factor: float = 1.0
    l1_bank_conflict_factor: float = 1.0
    simt_shm_conflict_factor: float = 1.0
    source_kind: str = "empty"

    @property
    def is_empty(self) -> bool:
        return self.source_kind == "empty" and self.total_transfer_bytes == 0 and self.total_vector_ops == 0 and self.cube_mmad_count == 0 and self.cube_macs_total == 0

    @property
    def total_vector_ops(self) -> int:
        return self.fp_alu_ops + self.int_alu_ops + self.sfu_f16_ops + self.sfu_f32_ops + self.reduce_ops + self.slot0_only_ops

    @property
    def total_transfer_bytes(self) -> int:
        return sum(getattr(self, name) for name in _COPY_EDGE_KEYS.values())

    @classmethod
    def coerce(cls, raw: MicroarchProfile | Mapping[str, object] | None) -> MicroarchProfile:
        """Coerce Mapping or MicroarchProfile (returning empty profile for None)."""
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(f"Expected MicroarchProfile or Mapping, got {type(raw).__name__}")
        kw: dict[str, Any] = {}
        for f in fields(cls):
            val = raw.get(f.name)
            if val is None:
                continue
            if f.name == "source_kind":
                kw[f.name] = str(val)
            elif f.name.endswith("_factor"):
                kw[f.name] = max(1.0, _safe_num(val, 1.0, True))
            elif f.name in _MIN1_FIELDS:
                kw[f.name] = max(1, _safe_num(val, 1))
            else:
                kw[f.name] = max(0, _safe_num(val, 0))
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


def _eval_extents(extents: Sequence[object], var_bindings: Mapping[str, int] | None) -> list[int] | None:
    dims: list[int] = []
    for ext in extents:
        iv = _eval_tirx_expr_int(ext, var_bindings)
        if iv is None or iv <= 0:
            return None
        dims.append(iv)
    return dims


def _map_tl_scope_name(raw_scope: object) -> str | None:
    return _TL_SCOPE_MAP.get(str(raw_scope or "").strip().lower())


def _extract_region_info(arg: object, *, tirx: Any, DataTypeCls: Any, var_bindings: Mapping[str, int] | None) -> tuple[str | None, int, int, str]:
    extents: Sequence[object] = ()
    explicit: int | None = None
    buf = getattr(arg, "buffer", None)
    if isinstance(arg, tirx.Call):
        op_name, args = str(getattr(arg.op, "name", arg.op)).lower(), list(arg.args)
        if ("region" in op_name or "access_ptr" in op_name) and args:
            buf = getattr(args[0], "buffer", None)
            extents = args[2:] if "region" in op_name else ()
            explicit = _eval_tirx_expr_int(args[3], var_bindings) if ("access_ptr" in op_name and len(args) > 3) else None
    if buf is None:
        return None, 0, 0, ""
    scope = _map_tl_scope_name(buf.scope()) if hasattr(buf, "scope") else None
    dtype_str = str(getattr(buf, "dtype", ""))
    elem_bits = _dtype_bits(dtype_str, DataTypeCls)
    dims = ([explicit] if (explicit and explicit > 0) else None) or (_eval_extents(extents, var_bindings) if extents else None) or _eval_extents(getattr(buf, "shape", ()) or (), var_bindings)
    elems = math.prod(dims) if dims else 0
    return scope, ((elems * elem_bits + 7) // 8 if elems > 0 else 0), max(0, elems), dtype_str


def _classify_dtype_family(dtype_str: str) -> str:
    low = str(dtype_str or "").strip().lower()
    if not low or low in ("void", "handle"):
        return "f32"
    _dtype_bytes_strict(low)
    if any(t in low for t in ("float16", "bfloat16")) or low.startswith(("f16", "bf16", "fp16")):
        return "f16"
    return "int" if ("int" in low or low == "bool" or low.startswith(("i", "u"))) else "f32"


def extract_microarch_profile_from_tirx(prim_func: object, *, var_bindings: Mapping[str, int] | None = None) -> MicroarchProfile:
    """Extract Ascend 950 MicroarchProfile from a TileLang PrimFunc or IRModule."""
    if prim_func is None:
        return MicroarchProfile()
    try:
        from tilelang import tvm  # noqa: PLC0415
        from tilelang.utils.language import DataType as TLDataType  # noqa: PLC0415
        from tvm import tirx  # noqa: PLC0415
    except ImportError:
        return MicroarchProfile()

    funcs = [f for f in prim_func.functions.values() if isinstance(f, tirx.PrimFunc)] if isinstance(prim_func, tvm.IRModule) else ([prim_func] if isinstance(prim_func, tirx.PrimFunc) else [])
    if not funcs:
        return MicroarchProfile()

    c: dict[str, int] = dict.fromkeys(
        ("fp_alu_ops", "int_alu_ops", "sfu_f16_ops", "sfu_f32_ops", "reduce_ops", "slot0_only_ops",
         "scalar_ops", "ub_load_bytes", "ub_store_bytes", "cube_mmad_count", "cube_macs_total",
         "cv_sync_barriers", "pipe_all_barriers", *_COPY_EDGE_KEYS.values()),
        0,
    )
    c.update({"pipeline_depth": 1, "inner_loop_trips": 1, "subkernel_count": len(funcs)})
    ub_total_bytes, ub_conflict_bytes, l1_buffers = 0, 0, []

    def _is_scalar(dtype_str: str) -> bool:
        low = str(dtype_str or "").strip().lower()
        if not low or low in ("void", "handle"):
            return True
        try:
            return int(TLDataType(low).lanes) <= 1
        except Exception:
            return "x" not in low

    def _ub_bytes(dtype_obj: object) -> int:
        try:
            dt = TLDataType(str(dtype_obj))
            return max(1, (int(dt.bits) * int(dt.lanes) + 7) // 8)
        except Exception:
            return max(1, int(math.ceil(_dtype_bytes_strict(dtype_obj))))

    def _record_buf(buf: object) -> None:
        nonlocal ub_total_bytes, ub_conflict_bytes
        scope = _map_tl_scope_name(buf.scope()) if hasattr(buf, "scope") else None
        dims = _eval_extents(getattr(buf, "shape", ()) or (), var_bindings) if scope else None
        if not dims:
            return
        bits = _dtype_bits(getattr(buf, "dtype", ""), TLDataType)
        total_b, inner_b = (math.prod(dims) * bits + 7) // 8, (dims[-1] * bits + 7) // 8
        if scope == "UB" and len(dims) >= 2 and math.prod(dims[:-1]) > 1:
            ub_total_bytes += total_b
            if inner_b >= 512 and inner_b % 512 == 0:
                ub_conflict_bytes += total_b
        elif scope == "L1":
            l1_buffers.append(total_b)

    def _scan_expr(expr: object, *, repeat: int, in_simtvf: bool = False) -> str:
        if expr is None:
            return ""
        dt_str = str(getattr(expr, "dtype", ""))
        if isinstance(expr, tirx.Call):
            op_name = str(getattr(expr.op, "name", expr.op)).lower()
            arg_dtypes = [_scan_expr(a, repeat=repeat, in_simtvf=in_simtvf) for a in expr.args]
            eff_dt = next((d for d in arg_dtypes if d), dt_str)
            fam = _classify_dtype_family(eff_dt)
            if any(t in op_name for t in ("vintlv", "vdintlv", "interleave")):
                c["slot0_only_ops"] += repeat
            elif any(t in op_name for t in ("exp", "log", "sqrt", "rsqrt", "div", "tanh", "erf", "sigmoid")):
                c["sfu_f16_ops" if fam == "f16" else "sfu_f32_ops"] += repeat
            elif "reduce" in op_name:
                c["reduce_ops"] += repeat
            elif any(t in op_name for t in ("simd", "vec", "cast", "fma")):
                c[("scalar_ops" if _is_scalar(eff_dt) else "int_alu_ops") if fam == "int" else "fp_alu_ops"] += repeat
            return eff_dt
        if isinstance(expr, tirx.BufferLoad):
            for idx_expr in getattr(expr, "indices", ()) or ():
                _scan_expr(idx_expr, repeat=repeat, in_simtvf=in_simtvf)
            if _map_tl_scope_name(expr.buffer.scope()) == "UB":
                c["ub_load_bytes"] += repeat * _ub_bytes(getattr(expr, "dtype", ""))
            return dt_str
        if isinstance(expr, (tirx.Add, tirx.Sub, tirx.Mul, tirx.Div, tirx.FloorDiv, tirx.Mod, tirx.Min, tirx.Max)):
            eff_dt = _scan_expr(expr.a, repeat=repeat, in_simtvf=in_simtvf) or _scan_expr(expr.b, repeat=repeat, in_simtvf=in_simtvf) or dt_str
            if _classify_dtype_family(eff_dt) == "int":
                scalar = isinstance(expr, (tirx.FloorDiv, tirx.Mod, tirx.Min, tirx.Max)) or _is_scalar(eff_dt) or in_simtvf
                c["scalar_ops" if scalar else "int_alu_ops"] += repeat
            else:
                c["fp_alu_ops"] += repeat
            return eff_dt
        return dt_str

    def _walk(stmt: object, *, repeat: int = 1, in_simtvf: bool = False) -> None:
        if stmt is None:
            return
        if isinstance(stmt, tirx.SeqStmt):
            for ch in stmt.seq:
                _walk(ch, repeat=repeat, in_simtvf=in_simtvf)
        elif isinstance(stmt, tirx.For):
            ann = getattr(stmt, "annotations", None) or {}
            raw_st = ann.get("num_stages", ann.get("software_pipeline_stage"))
            if raw_st is not None:
                c["pipeline_depth"] = max(c["pipeline_depth"], max(1, _safe_num(raw_st, 1)))
            if getattr(stmt, "thread_binding", None) is not None:
                _walk(stmt.body, repeat=repeat, in_simtvf=in_simtvf)
                return
            trip = _eval_tirx_expr_int(stmt.extent, var_bindings)
            kraw = getattr(stmt, "kind", 0)
            is_par = getattr(kraw, "value", kraw) == 1 or "parallel" in str(kraw).lower()
            eff = max(1, math.ceil(trip / 64)) if (in_simtvf and is_par and trip and trip > 0) else (trip if (trip and trip > 0) else 1)
            c["inner_loop_trips"] = max(c["inner_loop_trips"], eff)
            _walk(stmt.body, repeat=repeat * eff, in_simtvf=in_simtvf)
        elif isinstance(stmt, tirx.IfThenElse):
            _walk(stmt.then_case, repeat=repeat, in_simtvf=in_simtvf)
            _walk(stmt.else_case, repeat=repeat, in_simtvf=in_simtvf)
        elif isinstance(stmt, tirx.AttrStmt):
            ak = str(getattr(stmt, "attr_key", "")).lower()
            av = _safe_num(getattr(stmt, "value", None), 0)
            if "stage" in ak and av > 1:
                c["pipeline_depth"] = max(c["pipeline_depth"], av)
            _walk(stmt.body, repeat=repeat, in_simtvf=in_simtvf or ("simt" in ak) or ("simt" in str(getattr(stmt, "node", "")).lower()) or (ak == "pragma_parallel"))
        elif isinstance(stmt, tirx.SBlockRealize):
            _walk(stmt.block, repeat=repeat, in_simtvf=in_simtvf)
        elif isinstance(stmt, tirx.SBlock):
            for buf in getattr(stmt, "alloc_buffers", ()) or ():
                _record_buf(buf)
            ann = getattr(stmt, "annotations", None) or {}
            _walk(stmt.body, repeat=repeat, in_simtvf=in_simtvf or ("simt" in str(getattr(stmt, "name_hint", "")).lower()) or ("tl.vf_source_index" in ann))
        elif isinstance(stmt, (tirx.While, tirx.AllocBuffer, tirx.DeclBuffer, tirx.Bind)):
            if hasattr(stmt, "buffer"):
                _record_buf(stmt.buffer)
            if isinstance(stmt, tirx.Bind):
                _scan_expr(stmt.value, repeat=repeat, in_simtvf=in_simtvf)
            if hasattr(stmt, "body"):
                _walk(stmt.body, repeat=repeat, in_simtvf=in_simtvf)
        elif isinstance(stmt, tirx.BufferStore):
            for idx_expr in getattr(stmt, "indices", ()) or ():
                _scan_expr(idx_expr, repeat=repeat, in_simtvf=in_simtvf)
            if _map_tl_scope_name(stmt.buffer.scope()) == "UB":
                c["ub_store_bytes"] += repeat * _ub_bytes(getattr(stmt.value, "dtype", ""))
            _scan_expr(stmt.value, repeat=repeat, in_simtvf=in_simtvf)
        elif isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            op_name, args = str(getattr(stmt.value.op, "name", stmt.value.op)).lower(), list(stmt.value.args)
            if "copy" in op_name and len(args) >= 2:
                src_s, src_b, _, _ = _extract_region_info(args[0], tirx=tirx, DataTypeCls=TLDataType, var_bindings=var_bindings)
                dst_s, dst_b, _, _ = _extract_region_info(args[1], tirx=tirx, DataTypeCls=TLDataType, var_bindings=var_bindings)
                xfer = max(src_b, dst_b)
                edge = _COPY_EDGE_KEYS.get((src_s or "", dst_s or ""))
                if edge and xfer > 0:
                    c[edge] += repeat * xfer
                if src_s == "UB":
                    c["ub_load_bytes"] += repeat * xfer
                if dst_s == "UB":
                    c["ub_store_bytes"] += repeat * xfer
            elif ("gemm" in op_name or "mmad" in op_name) and len(args) >= 2:
                c["cube_mmad_count"] += repeat
                elems = [_extract_region_info(args[i], tirx=tirx, DataTypeCls=TLDataType, var_bindings=var_bindings)[2] for i in range(min(3, len(args)))]
                ba = getattr(args[0].args[0] if isinstance(args[0], tirx.Call) and args[0].args else args[0], "buffer", None)
                bb = getattr(args[1].args[0] if isinstance(args[1], tirx.Call) and args[1].args else args[1], "buffer", None)
                sa = _eval_extents((getattr(ba, "shape", ()) or ())[-2:], var_bindings) if ba else None
                sb = _eval_extents((getattr(bb, "shape", ()) or ())[-2:], var_bindings) if bb else None
                macs = (sa[0] * sa[1] * (sb[1] if sb[0] == sa[1] else sb[0])) if (sa and sb and len(sa) == 2 and len(sb) == 2) else max(elems, default=0) * 64
                c["cube_macs_total"] += repeat * macs
            elif "cross_flag" in op_name or "cross_core" in op_name:
                c["cv_sync_barriers"] += repeat
            elif "pipe_barrier" in op_name or "pipe_all" in op_name:
                c["pipe_all_barriers"] += repeat
            else:
                _scan_expr(stmt.value, repeat=repeat, in_simtvf=in_simtvf)

    for fn in funcs:
        for buf in (getattr(fn, "buffer_map", {}) or {}).values():
            _record_buf(buf)
        _walk(fn.body, repeat=1)

    ub_factor = (1.0 + 0.5 * (ub_conflict_bytes / float(ub_total_bytes))) if (ub_total_bytes > 0 and ub_conflict_bytes > 0) else 1.0
    l1_sum, l1_half = sum(l1_buffers), 256 * 1024
    l1_factor = 1.25 if l1_sum > l1_half else (1.12 if len(l1_buffers) >= 2 and l1_sum > l1_half // 2 else 1.0)
    return MicroarchProfile(**c, ub_bank_conflict_factor=round(ub_factor, 4), l1_bank_conflict_factor=round(l1_factor, 4), source_kind="tirx")


def extract_microarch_profile_from_cce(source_text: str) -> MicroarchProfile:
    """Extract unified post-codegen MicroarchProfile from Track A lowered CCE or Track B AscendC."""
    raw = str(source_text or "")
    if not raw.strip():
        return MicroarchProfile()
    clean = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", " ", raw, flags=re.DOTALL))
    sfu = re.findall(r"([^\n;]{0,80}\b(?:vexp|vln|vsqrt|vrsqrt|vdiv|Exp|Ln|Sqrt|Rsqrt|Div|Tanh|Sigmoid)\b[^\n;]{0,80})", clean, flags=re.IGNORECASE)
    sfu_f16 = sum(1 for s in sfu if any(t in s.lower() for t in ("half", "bfloat16", "fp16", "bf16", "__fp16")))
    xfers = {name: len(re.findall(pat, clean)) * 32768 for name, pat in _CCE_XFER_PATTERNS}
    simt_factor = 1.5 if (re.search(r"__simt_vf__|__shared__", clean) and any(int(c) in (16, 32, 64, 128) for _, c in re.findall(r"__shared__[^;]*\[(\d+)\]\s*\[(\d+)\]", clean))) else 1.0
    return MicroarchProfile(
        fp_alu_ops=len(re.findall(r"\b(?:vadd|vmul|vsub|vmad|vfma|vfadd|vfmul|Add|Mul|Sub|Muls|Adds)\b", clean)),
        int_alu_ops=len(re.findall(r"\b(?:viadd|vimul|vand|vor|vshl|vshr)\b", clean)),
        sfu_f16_ops=sfu_f16, sfu_f32_ops=len(sfu) - sfu_f16,
        reduce_ops=len(re.findall(r"\b(?:Reduce(?:Max|Min|Sum)|vcmax|vcmin|vcadd|reduce_(?:sum|max)|asc_reduce_\w+)\b", clean, flags=re.IGNORECASE)),
        slot0_only_ops=len(re.findall(r"\b(?:vintlv|vdintlv|Interleave|DeInterleave)\b", clean, flags=re.IGNORECASE)),
        ub_load_bytes=xfers["ub_to_l1_bytes"] + xfers["ub_to_gm_bytes"],
        ub_store_bytes=xfers["gm_to_ub_bytes"] + xfers["l0c_to_ub_bytes"],
        cube_mmad_count=len(re.findall(r"\b(?:Mmad|mmad|mad|cube::gemm|gemm_v\d|asc_mmad(?:_mx|_sparse)?|ascend_gemm_l1)\b\s*[(<]", clean)),
        cv_sync_barriers=len(re.findall(r"\b(?:CrossCore(?:Set|Wait)Flag|asc_sync_(?:notify|wait|block_arrive|block_wait))\b", clean)),
        pipe_all_barriers=len(re.findall(r"\b(?:pipe_barrier|asc_sync_pipe)\s*\(\s*PIPE_ALL\s*\)", clean)),
        subkernel_count=max(1, len(re.findall(r"\b__(?:aicore|simd_vf|simt_vf)__\b(?!\s*\()", clean))),
        simt_shm_conflict_factor=simt_factor, source_kind="cce", **xfers,
    )


def merge_microarch_profiles(stage_a: MicroarchProfile | Mapping[str, object] | None, stage_b: MicroarchProfile | Mapping[str, object] | None) -> MicroarchProfile:
    """Merge Stage-A TIR profile (exact byte volumes) with Stage-B CCE/AscendC profile."""
    pa, pb = MicroarchProfile.coerce(stage_a), MicroarchProfile.coerce(stage_b)
    if pa.is_empty:
        return pb
    if pb.is_empty:
        return pa
    merged: dict[str, Any] = {"source_kind": "merged"}
    for f in fields(MicroarchProfile):
        if f.name != "source_kind":
            av, bv = getattr(pa, f.name), getattr(pb, f.name)
            merged[f.name] = (av if av > 0 else bv) if f.name in _BYTE_FIELDS else max(av, bv)
    return MicroarchProfile(**merged)


__all__ = ["MicroarchProfile", "extract_microarch_profile_from_cce", "extract_microarch_profile_from_tirx", "merge_microarch_profiles"]

"""Native MLA builder.

Builds the current single-kernel MLA Module as native TIR: a symbolic job loop,
a symbolic KV-block loop and an execution scope. The builder carries the MLA
numerical contract (layout row mapping, GQA, right-aligned causal, V==K_nope,
rope-once, separate m/l and U recurrences, shard-derived grid and precision)
as typed metadata for the solver/storage/lowering stages.

No operator-side tilelang.compile, hand-written text kernel or legacy twin is
used. The lane-level body is an f32 transport template; the full mixed AIC/AIV
instruction schedule is a lowering obligation.
"""
from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.memory.alloc_tensor import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import DType, Layout, Mesh, TensorType, Topology
from tilefoundry.ir.types.dim import ceildiv
from tilefoundry.ir.types.utils import make_tensor_type
from tilefoundry_ascend import Ascend950PRTarget
from tilefoundry_ascend.ir.execution import SIMD_VF_KIND
from tilefoundry_ascend.ir.ops import GmToUb, Pset, UbToGm, Vld, Vmul, Vsts

BSND = "BSND"
BNSD = "BNSD"
LAYOUTS = (BSND, BNSD)
DTYPES = ("bfloat16", "float16")
PRECISION_SINGLE = "p_single"
PRECISION_HI_LO = "p_hi_lo"
PRECISION_POLICIES = (PRECISION_SINGLE, PRECISION_HI_LO)
LANES = 64


class MLABuildError(RuntimeError):
    """A violated MLA builder contract with a stable code."""

    def __init__(self, code: str, message: str, subject: str = "", expected=None, actual=None):
        self.code = code
        self.message = message
        self.subject = subject
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, message))

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "subject": self.subject,
            "expected": self.expected,
            "actual": self.actual,
        }


def _ceildiv_int(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class MLASpec:
    """The MLA contract consumed by the native builder."""

    batch: int
    seq_q: int
    seq_kv: int
    heads_q: int
    heads_kv: int
    d_nope: int
    d_rope: int
    layout: str = BSND
    dtype: str = "bfloat16"
    scale_value: float = -1.0
    is_causal: bool = True
    kv_block: int = 64
    block_q: int = 64
    shard_count: int = 1
    precision: str = PRECISION_SINGLE

    def __post_init__(self) -> None:
        for name, value in (
            ("batch", self.batch),
            ("seq_q", self.seq_q),
            ("seq_kv", self.seq_kv),
            ("heads_q", self.heads_q),
            ("heads_kv", self.heads_kv),
            ("d_nope", self.d_nope),
            ("d_rope", self.d_rope),
            ("kv_block", self.kv_block),
            ("block_q", self.block_q),
            ("shard_count", self.shard_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise MLABuildError("MLA_BAD_SHAPE", "%s must be an int >= 1" % name, name, ">= 1", value)
        if self.d_nope % 64:
            raise MLABuildError("MLA_BAD_D_NOPE", "d_nope must be a multiple of 64", "d_nope", "multiple of 64", self.d_nope)
        if self.d_rope != 64:
            raise MLABuildError("MLA_D_ROPE_MUST_BE_64", "d_rope must be 64", "d_rope", 64, self.d_rope)
        if self.heads_q % self.heads_kv:
            raise MLABuildError("MLA_NONDIVISIBLE_HEADS", "heads_q must be divisible by heads_kv", "heads", "Nq % Nkv == 0", "%d %% %d" % (self.heads_q, self.heads_kv))
        if self.layout not in LAYOUTS:
            raise MLABuildError("MLA_BAD_LAYOUT", "layout must be BSND or BNSD", "layout", LAYOUTS, self.layout)
        if self.dtype not in DTYPES:
            raise MLABuildError("MLA_BAD_DTYPE", "dtype must be bfloat16 or float16", "dtype", DTYPES, self.dtype)
        if self.precision not in PRECISION_POLICIES:
            raise MLABuildError("MLA_BAD_PRECISION", "precision must be p_single or p_hi_lo", "precision", PRECISION_POLICIES, self.precision)
        if self.is_causal and self.seq_q > self.seq_kv:
            raise MLABuildError("MLA_CAUSAL_NEEDS_S_LE_SKV", "right-aligned causal requires seq_q <= seq_kv", "seq", "S <= Skv", "S=%d Skv=%d" % (self.seq_q, self.seq_kv))
        if not isinstance(self.scale_value, (int, float)) or isinstance(self.scale_value, bool) or self.scale_value == 0:
            raise MLABuildError("MLA_BAD_SCALE", "scale_value must be finite and nonzero (<=0 selects 1/sqrt(D_qk))", "scale_value", "finite nonzero", self.scale_value)

    @property
    def group(self) -> int:
        return self.heads_q // self.heads_kv

    @property
    def d_qk(self) -> int:
        return self.d_nope + self.d_rope

    @property
    def d_v(self) -> int:
        return self.d_nope

    @property
    def scale(self) -> float:
        if self.scale_value is None or self.scale_value <= 0:
            return 1.0 / (self.d_qk ** 0.5)
        return float(self.scale_value)

    def kv_heads(self) -> int:
        return self.heads_kv

    def kv_blocks(self) -> int:
        return _ceildiv_int(self.seq_kv, self.kv_block)

    def query_row_position(self, row: int) -> tuple[int, int]:
        """Return (real query position i, GQA group g) for a flattened q row."""
        if not isinstance(row, int) or isinstance(row, bool) or row < 0 or row >= self.group * self.seq_q:
            raise MLABuildError("MLA_ROW_OUT_OF_RANGE", "query row out of range", "row", "0..G*S-1", row)
        if self.layout == BSND:
            i, g = divmod(row, self.group)
        else:
            g, i = divmod(row, self.seq_q)
        return i, g

    def query_head(self, row: int, kv_head: int = 0) -> int:
        _i, g = self.query_row_position(row)
        if not isinstance(kv_head, int) or isinstance(kv_head, bool) or not 0 <= kv_head < self.heads_kv:
            raise MLABuildError("MLA_KV_HEAD_OUT_OF_RANGE", "kv head out of range", "kv_head", "0..Nkv-1", kv_head)
        return kv_head * self.group + g

    def kv_head_of(self, query_head: int) -> int:
        if not isinstance(query_head, int) or isinstance(query_head, bool) or not 0 <= query_head < self.heads_q:
            raise MLABuildError("MLA_HEAD_OUT_OF_RANGE", "query head out of range", "query_head", "0..Nq-1", query_head)
        return query_head // self.group

    def causal_position_of(self, row: int) -> int:
        i, _g = self.query_row_position(row)
        return i

    def causal_limit(self, row: int) -> int:
        return self.causal_position_of(row) + (self.seq_kv - self.seq_q)

    def is_causal_masked(self, row: int, kv_index: int) -> bool:
        if not isinstance(kv_index, int) or isinstance(kv_index, bool) or kv_index < 0:
            raise MLABuildError("MLA_KV_INDEX_OUT_OF_RANGE", "kv index must be a nonnegative int", "kv_index", ">= 0", kv_index)
        return bool(self.is_causal and kv_index > self.causal_limit(row))

    def shard_ranges(self) -> tuple[tuple[int, int], ...]:
        base, extra = divmod(self.d_v, self.shard_count)
        ranges = []
        start = 0
        for shard in range(self.shard_count):
            width = base + (1 if shard < extra else 0)
            ranges.append((start, width))
            start += width
        return tuple(ranges)

    def program_count(self) -> int:
        row_tiles = _ceildiv_int(self.group * self.seq_q, self.block_q)
        return self.batch * self.heads_kv * row_tiles * self.shard_count

    def check_v_metadata(self, *, v_shape: tuple, v_dtype: str, v_layout: str, k_shape: tuple, k_dtype: str, k_layout: str) -> None:
        """Validate the V==K_nope metadata contract without reading tensor contents."""
        expected = (self.batch, self.seq_kv, self.heads_kv, self.d_nope) if self.layout == BSND else (self.batch, self.heads_kv, self.seq_kv, self.d_nope)
        if tuple(v_shape) != expected or tuple(k_shape) != expected:
            raise MLABuildError("MLA_V_SHAPE_MISMATCH", "V and K_nope must have the same contracted shape", "shape", expected, (tuple(v_shape), tuple(k_shape)))
        if v_dtype != self.dtype or k_dtype != self.dtype:
            raise MLABuildError("MLA_V_DTYPE_MISMATCH", "V and K_nope must have the declared dtype", "dtype", self.dtype, (v_dtype, k_dtype))
        if v_layout != self.layout or k_layout != self.layout:
            raise MLABuildError("MLA_V_LAYOUT_MISMATCH", "V and K_nope must have the declared layout", "layout", self.layout, (v_layout, k_layout))

    def contract(self) -> dict:
        return {
            "operator": "mla",
            "shape": {
                "batch": self.batch, "seq_q": self.seq_q, "seq_kv": self.seq_kv,
                "heads_q": self.heads_q, "heads_kv": self.heads_kv, "group": self.group,
                "d_nope": self.d_nope, "d_rope": self.d_rope, "layout": self.layout,
            },
            "dtype": self.dtype,
            "output_dtype": self.dtype,
            "row_mapping": {
                BSND: "r = i*G + g; hq = hkv*G + g",
                BNSD: "r = g*S + i; hq = hkv*G + g",
            },
            "gqa": "hkv = hq // G",
            "causal": "j <= i + (Skv - S), right-aligned; mask uses the decoded real i",
            "v_contract": "v == k_nope by value, not pointer alias; builder must not read v as a second GM tensor",
            "rope_added_once": "score = scale * (sum_nope + sum_rope); clear on first nope panel only, rope accumulated once",
            "recurrence": {"m_l": "m'=max(m,maxZ); l'=a*l+sum(P)", "U": "U'=a*U+P@V; U waits only for its own panels"},
            "store": "Y=U/l with safe denominator; fully masked/padded rows are not written to GM",
            "precision": self.precision,
            "denominator": "sum(P) in FP32; p_hi_lo denominator is not sum(P_hi+P_lo)",
            "shard": {"count": self.shard_count, "ranges": self.shard_ranges(), "valid_last_width": self.shard_ranges()[-1][1]},
            "grid": {"block_q": self.block_q, "program_count": self.program_count(), "source": "shard jobs plan_programs(program_axes), not undersharded wrapper cores"},
            "device_validated": False,
        }


def _alloc(var: Var, tensor_type: TensorType) -> LetStmt:
    return LetStmt(var=var, value=Call(target=AllocTensor(tensor_type=tensor_type), args=(), type=tensor_type))


def _lane_type(storage: str, dtype: DType = DType.f32) -> TensorType:
    return make_tensor_type((LANES,), dtype, storage)


def build_mla_module(spec: MLASpec) -> Module:
    """Return a native MLA Module with symbolic job/KV-block loops and AIV scope."""
    if not isinstance(spec, MLASpec):
        raise MLABuildError("MLA_BAD_SPEC", "build_mla_module expects MLASpec", "spec", "MLASpec", type(spec).__name__)
    target = Ascend950PRTarget()
    i64 = TensorType.scalar(DType.i64)
    gm = _lane_type("gmem")
    ub = _lane_type("smem")
    rmem = _lane_type("rmem")
    pred = _lane_type("rmem", DType.bool)

    q_nope = Var("q_nope_tile", type=gm)
    q_rope = Var("q_rope_tile", type=gm)
    k_nope = Var("k_nope_tile", type=gm)
    k_rope = Var("k_rope_tile", type=gm)
    v = Var("v_tile", type=gm)
    out = Var("out_tile", type=gm)
    job_count = Var("job_count", type=i64)
    kv_blocks = Var("kv_blocks", type=i64)
    job = Var("job", type=i64)
    kb = Var("kv_block", type=i64)

    ub_q = Var("ub_q", type=ub)
    ub_k = Var("ub_k", type=ub)
    ub_out = Var("ub_out", type=ub)
    va = Var("va", type=rmem)
    vb = Var("vb", type=rmem)
    vd = Var("vd", type=rmem)
    mask = Var("mask", type=pred)

    compute = ExecutionScope(
        kind=SIMD_VF_KIND,
        body=Sequential((
            _alloc(va, rmem),
            _alloc(vb, rmem),
            _alloc(vd, rmem),
            _alloc(mask, pred),
            Evaluate(callable=Pset(elem_width=32), args=(mask,)),
            Evaluate(callable=Vld(), args=(ub_q, va)),
            Evaluate(callable=Vld(), args=(ub_k, vb)),
            Evaluate(callable=Vmul(), args=(va, vb, mask, vd)),
            Evaluate(callable=Vsts(), args=(vd, ub_out)),
        )),
    )
    body = Sequential((
        _alloc(ub_q, ub),
        _alloc(ub_k, ub),
        _alloc(ub_out, ub),
        For(
            job,
            Constant(0, type=i64),
            job_count,
            Constant(1, type=i64),
            Sequential((
                Evaluate(callable=GmToUb(), args=(q_nope, ub_q)),
                For(
                    kb,
                    Constant(0, type=i64),
                    kv_blocks,
                    Constant(1, type=i64),
                    Sequential((
                        Evaluate(callable=GmToUb(), args=(k_nope, ub_k)),
                        compute,
                    )),
                ),
                Evaluate(callable=UbToGm(), args=(ub_out, out)),
            )),
        ),
    ))
    mesh = Mesh((Topology("aiv", 1),), Layout((1,)), ("core",))
    scoped = MeshScope(mesh, Var("core", type=mesh), body)
    # ABI ORDER. output_count counts TRAILING parameters, so out must be LAST. The old
    # (.., v, out, job_count, kv_blocks) order made the callable contract treat kv_blocks
    # as the output tensor and out as an input.
    function = PrimFunction(
        "mla",
        (q_nope, q_rope, k_nope, k_rope, v, job_count, kv_blocks, out),
        Sequential((scoped,)),
        output_count=1,
        target=target,
    )
    return Module(
        "mla_native",
        (function,),
        entry="mla",
        target=target,
        topologies=(Topology("aiv", 1),),
        metadata={
            "operator": "mla",
            "builder": "mla.build.build_mla_module",
            "native_ir": "tir.PrimFunction/For/ExecutionScope",
            "spec": spec,
            "contract": spec.contract(),
            "program_count": spec.program_count(),
            "device_validated": False,
            "lane_template": "f32 64-lane transport; full MIX AIC/AIV schedule and PACK_NZ scratch are lowering obligations",
        },
    )


def build_mla(spec: MLASpec) -> Module:
    """Short alias used by the mainline builder registry."""
    return build_mla_module(spec)


__all__ = [
    "BSND",
    "BNSD",
    "LAYOUTS",
    "DTYPES",
    "PRECISION_SINGLE",
    "PRECISION_HI_LO",
    "PRECISION_POLICIES",
    "MLABuildError",
    "MLASpec",
    "build_mla_module",
    "build_mla",
]

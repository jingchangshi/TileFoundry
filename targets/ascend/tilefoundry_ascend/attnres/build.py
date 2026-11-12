"""Native attention_residual builder.

This module builds the current native TIR Module for attention_residual. It does
not import a text template or call tilelang.compile. The body is a symbolic
whole-task skeleton: token-group and N-block loops are native For nodes, the
compute body is an ExecutionScope, and the numerical contract is carried as
typed metadata for the scheduler/storage/lowering stages.

The lane-level body is an f32 transport template. Storage dtype conversion and
the full score/merge body remain lowering obligations; no CPU/model result is
claimed as device validation.
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

ONLINE = "online"
BLOCKED_ONLINE = "blocked_online"
RESIDENT_N = "resident_n"
FAMILIES = (ONLINE, BLOCKED_ONLINE, RESIDENT_N)

LITERAL_FP32 = "literal_fp32"
FACTORED_GW = "factored_gw"
SCORE_MODES = (LITERAL_FP32, FACTORED_GW)

DTYPES = ("bfloat16", "float16")
TOKEN_ROWS = (1, 2, 4, 8, 16)
LEGAL_BLOCK_N = (1, 2, 4, 8, 16, 32, 64)
LANES = 64


class AttnResBuildError(RuntimeError):
    """A violated attention_residual builder contract with a stable code."""

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


@dataclass(frozen=True)
class AttnResSpec:
    """The author-visible attention_residual contract for the native builder."""

    n: int
    tokens: int
    d: int
    dtype: str = "bfloat16"
    token_rows: int = 1
    block_n: int = 1
    d_chunk: int = 1024
    v_versions: int = 1
    family: str = ONLINE
    score_mode: str = LITERAL_FP32

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("n", self.n, 2, 64),
            ("tokens", self.tokens, 1, 2**28),
            ("d", self.d, 256, 8192),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise AttnResBuildError(
                    "ATTNRES_BAD_DIMENSION",
                    "%s must be an int in [%d, %d]" % (name, minimum, maximum),
                    name,
                    "[%d, %d]" % (minimum, maximum),
                    value,
                )
        if self.dtype not in DTYPES:
            raise AttnResBuildError("ATTNRES_BAD_DTYPE", "dtype must be bfloat16 or float16", "dtype", DTYPES, self.dtype)
        if self.token_rows not in TOKEN_ROWS:
            raise AttnResBuildError("ATTNRES_BAD_TOKEN_ROWS", "token_rows is outside the legal set", "token_rows", TOKEN_ROWS, self.token_rows)
        if self.family not in FAMILIES:
            raise AttnResBuildError("ATTNRES_BAD_FAMILY", "unknown family", "family", FAMILIES, self.family)
        if self.score_mode not in SCORE_MODES:
            raise AttnResBuildError("ATTNRES_BAD_SCORE_MODE", "unknown score mode", "score_mode", SCORE_MODES, self.score_mode)
        legal = set(LEGAL_BLOCK_N) | {self.n}
        if not isinstance(self.block_n, int) or isinstance(self.block_n, bool) or self.block_n not in legal or not 1 <= self.block_n <= self.n:
            raise AttnResBuildError(
                "ATTNRES_BAD_BLOCK_N",
                "block_n must be in the legal CN set and no larger than N",
                "block_n",
                tuple(sorted(value for value in legal if value <= self.n)),
                self.block_n,
            )
        if self.family == ONLINE and self.block_n != 1:
            raise AttnResBuildError("ATTNRES_FAMILY_CN_CONTRADICTION", "online requires block_n=1", "block_n", 1, self.block_n)
        if self.family == RESIDENT_N and self.block_n != self.n:
            raise AttnResBuildError("ATTNRES_FAMILY_CN_CONTRADICTION", "resident_n requires block_n=N", "block_n", self.n, self.block_n)
        if self.family == BLOCKED_ONLINE and not 1 < self.block_n < self.n:
            raise AttnResBuildError(
                "ATTNRES_FAMILY_CN_CONTRADICTION",
                "blocked_online requires 1 < block_n < N",
                "block_n",
                "1 < CN < N",
                self.block_n,
            )
        if not isinstance(self.d_chunk, int) or isinstance(self.d_chunk, bool) or not 1 <= self.d_chunk <= self.d:
            raise AttnResBuildError("ATTNRES_BAD_D_CHUNK", "d_chunk must be in [1, D]", "d_chunk", "[1, D]", self.d_chunk)
        if not isinstance(self.v_versions, int) or isinstance(self.v_versions, bool) or self.v_versions not in (1, 2, 3):
            raise AttnResBuildError("ATTNRES_BAD_V_VERSIONS", "v_versions must be 1, 2 or 3", "v_versions", (1, 2, 3), self.v_versions)
        if self.n * self.tokens * self.d > 2**28:
            raise AttnResBuildError("ATTNRES_DOMAIN_TOO_LARGE", "N*M*D exceeds 2**28", "N*M*D", "<= 2**28", self.n * self.tokens * self.d)

    def n_blocks(self) -> int:
        return (self.n + self.block_n - 1) // self.block_n

    def groups(self) -> int:
        return (self.tokens + self.token_rows - 1) // self.token_rows

    def padded_d(self) -> int:
        value = 1
        while value < self.d:
            value *= 2
        return value

    def contract(self) -> dict:
        return {
            "operator": "attention_residual",
            "shape": {"N": self.n, "M": self.tokens, "D": self.d, "PD": self.padded_d()},
            "dtype": self.dtype,
            "output_dtype": self.dtype,
            "families": {
                ONLINE: "CN=1; update running m/l/U after every V block",
                BLOCKED_ONLINE: "1<CN<N; update after each CN-block group",
                RESIDENT_N: "CN=N; one running-state update, all V resident",
            },
            "reductions": {"rms_square_sum": "D", "score_dot": "D", "softmax": "N", "weighted_sum": "N"},
            "fp32_state": ("m", "l", "acc", "score"),
            "score_modes": {
                LITERAL_FP32: "rms = sqrt(mean(v^2)+eps); score = sum_d((v/rms)*gamma*w)/sqrt(D)",
                FACTORED_GW: "gw=gamma*w; q=sum_d(v^2); dot=sum_d(v*gw); rms=sqrt(q/D+eps); score=dot/(rms*sqrt(D))",
            },
            "residual": "y = (1-alpha)*x + alpha*weighted_sum(original V)",
            "runtime_scalars": ("alpha", "epsilon"),
            "tail": "N tail uses n<N weight zero; D tail uses i<D mask before compute; PD is allocation only",
            "pipeline": {"token_rows": self.token_rows, "block_n": self.block_n, "d_chunk": self.d_chunk, "v_versions": self.v_versions},
            "device_validated": False,
        }


def _alloc(var: Var, tensor_type: TensorType) -> LetStmt:
    return LetStmt(var=var, value=Call(target=AllocTensor(tensor_type=tensor_type), args=(), type=tensor_type))


def _lane_type(storage: str, dtype: DType = DType.f32) -> TensorType:
    return make_tensor_type((LANES,), dtype, storage)


def build_attnres_module(spec: AttnResSpec) -> Module:
    """Return a native Module with symbolic group/N-block loops and an AIV scope."""
    if not isinstance(spec, AttnResSpec):
        raise AttnResBuildError("ATTNRES_BAD_SPEC", "build_attnres_module expects AttnResSpec", "spec", "AttnResSpec", type(spec).__name__)
    target = Ascend950PRTarget()
    i64 = TensorType.scalar(DType.i64)
    gm = _lane_type("gmem")
    ub = _lane_type("smem")
    rmem = _lane_type("rmem")
    pred = _lane_type("rmem", DType.bool)

    x = Var("x_tile", type=gm)
    v = Var("v_tile", type=gm)
    y = Var("y_tile", type=gm)
    m_extent = Var("M_extent", type=i64)
    n_extent = Var("N_extent", type=i64)
    group = Var("group", type=i64)
    block = Var("block", type=i64)

    ub_x = Var("ub_x", type=ub)
    ub_v = Var("ub_v", type=ub)
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
            Evaluate(callable=Vld(), args=(ub_v, va)),
            Evaluate(callable=Vld(), args=(ub_x, vb)),
            Evaluate(callable=Vmul(), args=(va, vb, mask, vd)),
            Evaluate(callable=Vsts(), args=(vd, ub_out)),
        )),
    )
    body = Sequential((
        _alloc(ub_x, ub),
        _alloc(ub_v, ub),
        _alloc(ub_out, ub),
        For(
            group,
            Constant(0, type=i64),
            ceildiv(m_extent, spec.token_rows),
            Constant(1, type=i64),
            Sequential((
                Evaluate(callable=GmToUb(), args=(x, ub_x)),
                For(
                    block,
                    Constant(0, type=i64),
                    ceildiv(n_extent, spec.block_n),
                    Constant(1, type=i64),
                    Sequential((
                        Evaluate(callable=GmToUb(), args=(v, ub_v)),
                        compute,
                    )),
                ),
                Evaluate(callable=UbToGm(), args=(ub_out, y)),
            )),
        ),
    ))
    mesh = Mesh((Topology("aiv", 1),), Layout((1,)), ("core",))
    scoped = MeshScope(mesh, Var("core", type=mesh), body)
    function = PrimFunction(
        "attention_residual",
        (x, v, y, m_extent, n_extent),
        Sequential((scoped,)),
        output_count=1,
        target=target,
    )
    return Module(
        "attnres_native",
        (function,),
        entry="attention_residual",
        target=target,
        topologies=(Topology("aiv", 1),),
        metadata={
            "operator": "attention_residual",
            "builder": "attnres.build.build_attnres_module",
            "native_ir": "tir.PrimFunction/For/ExecutionScope",
            "spec": spec,
            "contract": spec.contract(),
            "device_validated": False,
            "lane_template": "f32 64-lane transport; storage dtype conversion and full score/merge body are lowering obligations",
        },
    )


def build_attnres(spec: AttnResSpec) -> Module:
    """Short alias used by the mainline builder registry."""
    return build_attnres_module(spec)


__all__ = [
    "ONLINE",
    "BLOCKED_ONLINE",
    "RESIDENT_N",
    "FAMILIES",
    "LITERAL_FP32",
    "FACTORED_GW",
    "SCORE_MODES",
    "DTYPES",
    "TOKEN_ROWS",
    "AttnResBuildError",
    "AttnResSpec",
    "build_attnres_module",
    "build_attnres",
]

"""V02 (03-04..03-09): carrier / vcvt typed contract, positives and named refusals.

Three layers are used, and each refusal records WHICH layer stated it:

  * typed admission -- check_call(op, TensorTypes, mesh), the SAME checker the HIR
    type inferencer and the TIR walker call;
  * authored HIR -- a real .py fixture through the public CLI 'analyze';
  * reloaded TIR -- a hand-built direct TIR, or an edited copy of the PUBLIC FINALIZE
    artifact reloaded with load_authored_ir, then verify_module/verify_prim_function.

A refusal is only evidence when it carries the rule's own wording, so every negative
asserts a message fragment. A TypeError from Op.__init__ is NOT a verdict: 'part' is
optional at construction and refused BY NAME by _VcvtSignature instead.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR. --group selects one WBS item's checks.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import tilefoundry_ascend  # noqa: F401

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.schedule import ScheduleRefusal
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import DType, Mesh, StorageKind, Topology, make_tensor_type
from tilefoundry.visitor_registry.access_relation import identity_relations, register_access_relation
from tilefoundry.visitor_registry.instruction import check_call
from tilefoundry.visitor_registry.verify import (
    verify_module,
    verify_predicate_producers,
    verify_prim_function,
)
from tilefoundry_ascend.ir.execution import SIMD_VF_DECLARATION
from tilefoundry_ascend.ir.ops import Pset, Pset16, Vadd, Vcvt, Vld, Vsts

if "torch_npu" in sys.modules:
    raise SystemExit("REFUSED: torch_npu is loaded; this check is CPU-only")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "p3_carrier")
CHAIN_TIR = os.environ.get("P3_CHAIN_TIR", "")
KIND, LANES = "ascend.simdvf", 64
AIV = Mesh((Topology("aiv", 1),), (1,), ("core",))

VCVT_SIGNATURE = "ascend_vcvt: src="
GRANULARITY = "requires a 16-bit mask produced by a pset of the same granularity"
NOT_A_PRODUCER = "which is not a declared mask producer"
NO_WIDTH = "at None-bit mask granularity"
DOMINANCE = "has no producing instruction before it"
LANES_RULE = "is not the 64 lanes execution region"
PLACEMENT_WIDTH = "declares 16 for conversion instructions"
CARRIER_BOUNDARY = "carrier is private to execution region"


def cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "tilefoundry.cli", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def ty(dtype, storage, shape=(LANES,)):
    return make_tensor_type(shape, dtype=DType.from_name(dtype), storage=storage)


def bind(name, type_):
    var = Var(name=name, type=type_)
    return LetStmt(var=var, value=Call(target=AllocTensor(tensor_type=type_), args=(), type=type_)), var


def function(*body):
    return PrimFunction(
        name="fn", params=(),
        body=MeshScope(AIV, Var(name="aiv", type=ty("f32", "smem")), Sequential(body=body)),
        output_count=0,
    )


def admission(src_dtype, src_storage, dst_dtype, dst_storage, **attrs):
    """None when admitted, otherwise the refusal text."""
    try:
        op = Vcvt(**attrs)
        check_call(op, (ty(src_dtype, src_storage), ty("bool", "rmem"), ty(dst_dtype, dst_storage)), AIV, frame=None)
    except ScheduleRefusal as error:
        return str(error)
    return None


def expect_refusal(message, needle, what):
    if message is None:
        raise AssertionError(f"{what}: ACCEPTED; the checked rule did not fire")
    if needle not in message:
        raise AssertionError(f"{what}: refused, but not by the expected rule ({needle!r}): {message[:300]}")
    lines = [line.strip() for line in message.strip().splitlines() if needle in line]
    print("       " + (lines[0] if lines else message.strip().splitlines()[-1])[:190])


FWD = dict(direction="forward", part=0, elem_width=16)
REV = dict(direction="reverse", part=0, elem_width=16)


# ------------------------------ positives ------------------------------------

def check_forward_signature_admitted():
    result = admission("bf16", "carrier", "f32", "rmem", **FWD)
    if result is not None:
        raise AssertionError(f"forward carrier->f32 refused: {result}")


def check_reverse_signature_admitted():
    """f32 is NOT blanket-refused: the reverse direction takes an f32 RMEM source."""
    result = admission("f32", "rmem", "bf16", "carrier", **REV)
    if result is not None:
        raise AssertionError(f"reverse f32->carrier refused: {result}")


def check_mode_unchanged():
    if tuple(SIMD_VF_DECLARATION.element_dtypes) != (DType.f32,):
        raise AssertionError("element_dtypes widened")
    if SIMD_VF_DECLARATION.mask_width_bits != 32 or SIMD_VF_DECLARATION.conversion_mask_width_bits != 16:
        raise AssertionError("mask widths changed")
    if Vcvt.mask_class != "conversion" or Vcvt.mask_width_attribute != "elem_width":
        raise AssertionError("vcvt mask class/width declaration is not conversion/16")
    if Pset16.mask_class != "conversion" or getattr(Pset, "mask_class", "arithmetic") == "conversion":
        raise AssertionError("pset32/pset16 classes are not separated")


def check_public_chain_pipeline():
    """Public analyze/finalize/reload/verify on the authored bf16->f32 chain."""
    if not CHAIN_TIR or not os.path.isfile(CHAIN_TIR):
        raise AssertionError("P3_CHAIN_TIR must name the public finalize artifact")
    reloaded = load_authored_ir(f"{CHAIN_TIR}:Bf16VcvtChain")
    verify_module(reloaded)
    verify_prim_function(reloaded.entry_function())
    text = open(CHAIN_TIR, encoding="utf-8").read()
    for needle in ('"carrier"', 'dist="UNPK_B16"', "ascend_pset16", "elem_width=16", 'direction="forward"', "part=0", "ascend_vsts"):
        if needle not in text:
            raise AssertionError(f"finalized TIR lost {needle!r}")
    print("       public TIR reloaded and verified; carrier/pset16/vcvt/vsts retained")


def check_f32_legacy_positive():
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "p3_legacy_vadd.py")
    result = cli("schedule", "finalize", os.path.join(ROOT, "examples", "rmsnorm_950pr", "vf_vadd.py") + ":VfVadd", out)
    if result.returncode != 0:
        raise AssertionError(f"f32 pset32/vadd finalize regressed: {result.stderr[-300:]}")


# --------------------------- direct-TIR helpers ------------------------------

def chain_tir(*, producer=Pset16, width=16, order="before", part=0, elem_width=16,
              direction="forward", carrier_shape=(LANES,), src_var="carrier", producer_cls=None):
    ub_s, ub = bind("ub", ty("bf16", "smem"))
    c_s, c = bind("carrier", ty("bf16", "carrier", carrier_shape))
    m_s, m = bind("mask", ty("bool", "rmem"))
    f_s, f = bind("f32", ty("f32", "rmem"))
    plain_s, plain = bind("plain", ty("f32", "rmem"))
    out_s, out = bind("out", ty("f32", "smem"))
    inner = [ub_s, c_s, m_s, f_s, plain_s, out_s,
             Evaluate(callable=Vld(dist="UNPK_B16"), args=(ub, c))]
    cls = producer_cls or producer
    pset = Evaluate(callable=cls(elem_width=width), args=(m,))
    if order == "before":
        inner.append(pset)
    source = c if src_var == "carrier" else plain
    inner.append(Evaluate(callable=Vcvt(direction=direction, part=part, elem_width=elem_width), args=(source, m, f)))
    if order == "after":
        inner.append(pset)
    if order == "none":
        pass
    inner.append(Evaluate(callable=Vsts(), args=(f, out)))
    return function(ExecutionScope(kind=KIND, body=Sequential(body=tuple(inner))))


class _UndeclaredProducer(Pset):
    """Test-only: a bool writer that is NOT a declared mask producer."""
    mask_width_attribute = None


class _WidthlessProducer(Pset):
    """Test-only: a bool writer whose declared width attribute resolves to None."""
    mask_width_attribute = "no_such_attribute"


for _cls in (_UndeclaredProducer, _WidthlessProducer):
    try:
        register_access_relation(_cls)(identity_relations)
    except RuntimeError:
        pass


def tir_refusal(fn):
    try:
        verify_prim_function(fn)
    except Exception as error:  # noqa: BLE001 -- the message is the verdict
        return str(error)
    return None


def vadd_tir(mask_cls, width):
    ub_s, ub = bind("ub", ty("f32", "smem"))
    a_s, a = bind("a", ty("f32", "rmem"))
    b_s, b = bind("b", ty("f32", "rmem"))
    m_s, m = bind("mask", ty("bool", "rmem"))
    d_s, d = bind("acc", ty("f32", "rmem"))
    inner = [ub_s, a_s, b_s, m_s, d_s,
             Evaluate(callable=Vld(), args=(ub, a)),
             Evaluate(callable=Vld(), args=(ub, b)),
             Evaluate(callable=mask_cls(elem_width=width), args=(m,)),
             Evaluate(callable=Vadd(), args=(a, b, m, d))]
    return function(ExecutionScope(kind=KIND, body=Sequential(body=tuple(inner))))


# ------------------------------ negatives ------------------------------------

def check_carrier_into_arithmetic_admission():
    try:
        check_call(Vadd(), (ty("bf16", "carrier"), ty("f32", "rmem"), ty("bool", "rmem"), ty("f32", "rmem")), AIV, frame=None)
    except ScheduleRefusal as error:
        text = str(error)
        if "T.ascend_vadd" not in text or "does not match f32" not in text:
            raise AssertionError(f"carrier->vadd refused by the wrong layer: {text}")
        print("       " + text.strip().splitlines()[-1][:170])
    else:
        raise AssertionError("carrier into vadd was ACCEPTED")


def check_carrier_into_arithmetic_tir():
    c_s, c = bind("carrier", ty("bf16", "carrier"))
    ub_s, ub = bind("ub", ty("f32", "smem"))
    b_s, b = bind("b", ty("f32", "rmem"))
    m_s, m = bind("mask", ty("bool", "rmem"))
    d_s, d = bind("acc", ty("f32", "rmem"))
    inner = [c_s, ub_s, b_s, m_s, d_s,
             Evaluate(callable=Pset(elem_width=32), args=(m,)),
             Evaluate(callable=Vadd(), args=(c, b, m, d))]
    fn = function(ExecutionScope(kind=KIND, body=Sequential(body=tuple(inner))))
    text = tir_refusal(fn)
    expect_refusal(text, "T.ascend_vadd", "carrier->vadd (TIR)")


def check_carrier_into_element_register():
    """vsts is an element-register store: a carrier is not an f32 element register."""
    try:
        check_call(Vsts(), (ty("bf16", "carrier"), ty("f32", "smem")), AIV, frame=None)
    except ScheduleRefusal as error:
        text = str(error)
        if "T.ascend_vsts" not in text or "does not match f32" not in text:
            raise AssertionError(f"carrier->vsts refused by the wrong layer: {text}")
        print("       " + text.strip().splitlines()[-1][:170])
    else:
        raise AssertionError("carrier into vsts was ACCEPTED")


def check_carrier_outside_region():
    c_s, c = bind("carrier", ty("bf16", "carrier"))
    ub_s, ub = bind("ub", ty("bf16", "smem"))
    scope = ExecutionScope(kind=KIND, body=Sequential(body=(Evaluate(callable=Vld(dist="UNPK_B16"), args=(ub, c)),)))
    text = tir_refusal(function(c_s, ub_s, scope))
    expect_refusal(text, CARRIER_BOUNDARY, "carrier outside region")


def check_carrier_used_after_region():
    """The private carrier Var must not escape its ExecutionScope (binder layer)."""
    c_s, c = bind("carrier", ty("bf16", "carrier"))
    ub_s, ub = bind("ub", ty("bf16", "smem"))
    out_s, out = bind("out", ty("f32", "smem"))
    scope = ExecutionScope(kind=KIND, body=Sequential(body=(c_s, Evaluate(callable=Vld(dist="UNPK_B16"), args=(ub, c)))))
    text = tir_refusal(function(ub_s, out_s, scope, Evaluate(callable=Vsts(), args=(c, out))))
    expect_refusal(text, "used outside the scope that binds it", "carrier escape")


def check_forward_on_plain_f32():
    text = admission("f32", "rmem", "f32", "rmem", **FWD)
    expect_refusal(text, VCVT_SIGNATURE, "forward vcvt on plain f32")


def check_forward_on_plain_f32_tir():
    expect_refusal(tir_refusal(chain_tir(src_var="plain")), VCVT_SIGNATURE, "forward vcvt on plain f32 (TIR)")


def check_wrong_producer_grain_to_vcvt():
    """Same bool shape, only the PRODUCER's declared granularity changes: 32 -> 16."""
    expect_refusal(tir_refusal(chain_tir(producer=Pset, width=32)), GRANULARITY, "pset32 -> vcvt16")


def check_wrong_producer_grain_to_vadd():
    """And the other way: a 16-bit producer feeding FP32 arithmetic."""
    text = tir_refusal(vadd_tir(Pset16, 16))
    expect_refusal(text, "requires a 32-bit mask", "pset16 -> vadd32")


def check_consumer_width_32():
    expect_refusal(tir_refusal(chain_tir(elem_width=32)), PLACEMENT_WIDTH, "vcvt elem_width=32")


def check_admission_width_32():
    expect_refusal(admission("bf16", "carrier", "f32", "rmem", **{**FWD, "elem_width": 32}), VCVT_SIGNATURE, "vcvt width 32 (admission)")


def check_pset32_vadd_positive():
    if tir_refusal(vadd_tir(Pset, 32)) is not None:
        raise AssertionError("the old f32 pset32/vadd positive was refused")


def check_part_missing():
    expect_refusal(admission("bf16", "carrier", "f32", "rmem", direction="forward", elem_width=16), "part=None", "part missing")


def check_part_one():
    expect_refusal(admission("bf16", "carrier", "f32", "rmem", **{**FWD, "part": 1}), VCVT_SIGNATURE, "part=1")


def check_part_bool():
    expect_refusal(admission("bf16", "carrier", "f32", "rmem", **{**FWD, "part": True}), VCVT_SIGNATURE, "part=True")


def check_part_wrong_type():
    text = admission("bf16", "carrier", "f32", "rmem", **{**FWD, "part": "0"})
    expect_refusal(text, "part must be int, got str", "part wrong type")


def check_direction_unknown():
    expect_refusal(admission("bf16", "carrier", "f32", "rmem", **{**FWD, "direction": "sideways"}), VCVT_SIGNATURE, "direction unknown")


def check_carrier_layout_wrong():
    expect_refusal(tir_refusal(chain_tir(carrier_shape=(32,))), LANES_RULE, "carrier extent 32")


def check_late_pset():
    expect_refusal(tir_refusal(chain_tir(order="after")), DOMINANCE, "pset after vcvt")


def check_no_pset():
    expect_refusal(tir_refusal(chain_tir(order="none")), DOMINANCE, "no pset")


def check_non_pset_bool_writer():
    expect_refusal(tir_refusal(chain_tir(producer_cls=_UndeclaredProducer, width=32)), NOT_A_PRODUCER, "non-pset bool writer")


def check_widthless_producer():
    expect_refusal(tir_refusal(chain_tir(producer_cls=_WidthlessProducer, width=32)), NO_WIDTH, "producer with no declared width")


def check_same_kind_different_instance():
    """DIRECT HELPER layer: the producer set is per ExecutionScope INSTANCE.

    A bool RMEM mask cannot be allocated outside a mode (that is refused earlier by
    verify_allocation_placement), and a Var bound in scope A is not visible in scope B
    (binder layer). So a full well-typed PrimFunction cannot express this case without
    an earlier refusal. The architect's gate asks for the helper-layer constructive
    case instead: call verify_predicate_producers on each scope directly. That is the
    function the public verify path calls once per ExecutionScope (verify.py:636), and
    its fresh-set-per-call property is exactly "same kind, different instance".
    """
    m_s, m = bind("mask", ty("bool", "rmem"))
    c_s, c = bind("carrier", ty("bf16", "carrier"))
    f_s, f = bind("f32", ty("f32", "rmem"))
    ub_s, ub = bind("ub", ty("bf16", "smem"))
    producer = Evaluate(callable=Pset16(elem_width=16), args=(m,))
    consumer = Evaluate(callable=Vcvt(**FWD), args=(c, m, f))
    scope_a = ExecutionScope(kind=KIND, body=Sequential(body=(m_s, producer)))
    scope_b = ExecutionScope(kind=KIND, body=Sequential(body=(c_s, f_s, ub_s, consumer)))
    scope_ok = ExecutionScope(kind=KIND, body=Sequential(body=(m_s, c_s, f_s, ub_s, producer, consumer)))
    verify_predicate_producers(scope_a)   # A has its own producer, no reader
    verify_predicate_producers(scope_ok)  # positive control: same instance, producer first
    try:
        verify_predicate_producers(scope_b)
    except Exception as error:  # noqa: BLE001 -- the message is the verdict
        expect_refusal(str(error), DOMINANCE, "same kind, different instance (helper)")
    else:
        raise AssertionError("scope B accepted a mask produced only in sibling scope A")


def check_public_pipeline_negative_from_reload(path_needle, fragment, what):
    """Reload an EDITED copy of the public finalize artifact and refuse it by name."""
    if not CHAIN_TIR or not os.path.isfile(CHAIN_TIR):
        raise AssertionError("P3_CHAIN_TIR must name the public finalize artifact")
    text = open(CHAIN_TIR, encoding="utf-8").read()
    edited = path_needle(text)
    if edited == text:
        raise AssertionError(f"{what}: reload mutation did not change the TIR text")
    directory = os.environ.get("TMPDIR", "/tmp")
    path = os.path.join(directory, f"p3_reload_{what}.py")
    open(path, "w", encoding="utf-8").write(edited)
    try:
        module = load_authored_ir(f"{path}:Bf16VcvtChain")
        verify_module(module)
        refusal = tir_refusal(module.entry_function())
    except Exception as error:  # noqa: BLE001 -- reload/verify refusal is the verdict
        refusal = str(error)
    expect_refusal(refusal, fragment, what)


# --------------------- authored HIR and reloaded TIR -------------------------

CHAIN_SRC = os.path.join(FIX, "bf16_vcvt_chain.py")
PSET16_LINE = "                pred16 = tf.schedule((mask16,), op=T.ascend_pset16(elem_width=16))"
VCVT_LINE = "                converted = tf.schedule("
TIR_PSET16 = "                T.ascend_pset16(mask16, elem_width=16)"
TIR_VCVT = "                T.ascend_vcvt(carrier, mask16, f32reg, direction=\"forward\", part=0, elem_width=16)"


def _scratch(name, text):
    directory = os.environ.get("TMPDIR", "/tmp")
    path = os.path.join(directory, name)
    open(path, "w", encoding="utf-8").write(text)
    return path


def hir_analyze(mutation, what):
    """Run public CLI analyze on an authored HIR fixture derived from the chain."""
    original = open(CHAIN_SRC, encoding="utf-8").read()
    edited = mutation(original)
    if edited == original:
        raise AssertionError(f"{what}: HIR mutation did not change the source")
    path = _scratch(f"p3_hir_{what}.py", edited)
    result = cli("analyze", f"{path}:Bf16VcvtChain")
    return result.returncode, result.stderr


def check_hir_part_refused():
    code, stderr = hir_analyze(lambda t: t.replace("part=0, elem_width=16", "part=1, elem_width=16"), "part1")
    if code == 0:
        raise AssertionError("HIR accepted part=1")
    expect_refusal(stderr, VCVT_SIGNATURE, "HIR part=1")


def check_hir_part_missing_refused():
    code, stderr = hir_analyze(lambda t: t.replace("part=0, elem_width=16", "elem_width=16"), "part_missing")
    if code == 0:
        raise AssertionError("HIR accepted a missing part")
    expect_refusal(stderr, "part=None", "HIR part missing")


def check_hir_plain_f32_refused():
    code, stderr = hir_analyze(lambda t: t.replace("(loaded, pred16, f32reg)", "(f32reg, pred16, f32reg)"), "plain_f32")
    if code == 0:
        raise AssertionError("HIR accepted forward vcvt on a plain f32 register")
    expect_refusal(stderr, VCVT_SIGNATURE, "HIR plain f32")


def check_hir_width32_refused():
    code, stderr = hir_analyze(lambda t: t.replace("part=0, elem_width=16", "part=0, elem_width=32"), "width32")
    if code == 0:
        raise AssertionError("HIR accepted elem_width=32 on a conversion")
    expect_refusal(stderr, VCVT_SIGNATURE, "HIR width 32")


def check_hir_producer_negatives_are_tir_layer():
    """HIR has no producer graph: these two are ACCEPTED at analyze and refused at TIR.

    This is a layer fact, not a pass: 'analyze exit 0' is recorded as the HIR result and
    the corresponding reload-TIR negative (below) is the one that refuses by name.
    """
    code, stderr = hir_analyze(lambda t: t.replace("T.ascend_pset16(elem_width=16)", "T.ascend_pset(elem_width=32)"), "pset32")
    if code != 0:
        print(f"       HIR pset32 refused at analyze: {stderr.strip()[:120]}")
    else:
        print("       HIR pset32 -> vcvt ACCEPTED at analyze (no producer graph); refusal is at TIR")
    code, stderr = hir_analyze(
        lambda t: t.replace(PSET16_LINE + chr(10), "").replace(
            "                stored = tf.empty", PSET16_LINE + chr(10) + "                stored = tf.empty"),
        "late_pset")
    if code != 0:
        print(f"       HIR late pset refused at analyze: {stderr.strip()[:120]}")
    else:
        print("       HIR late pset ACCEPTED at analyze (no dominance walk); refusal is at TIR")


def check_reload_pset32_refused():
    check_public_pipeline_negative_from_reload(
        lambda t: t.replace(TIR_PSET16, "                T.ascend_pset(mask16, elem_width=32)"),
        GRANULARITY, "reload_pset32")


def check_reload_no_pset_refused():
    check_public_pipeline_negative_from_reload(
        lambda t: t.replace(TIR_PSET16 + chr(10), ""), DOMINANCE, "reload_no_pset")


def check_reload_late_pset_refused():
    def swap(t):
        if TIR_PSET16 not in t or TIR_VCVT not in t:
            raise AssertionError("reload mutation: pset/vcvt lines not found in public TIR")
        return t.replace(TIR_PSET16 + chr(10), "").replace(TIR_VCVT, TIR_VCVT + chr(10) + TIR_PSET16)
    check_public_pipeline_negative_from_reload(swap, DOMINANCE, "reload_late_pset")


def check_reload_part1_refused():
    check_public_pipeline_negative_from_reload(
        lambda t: t.replace("part=0, elem_width=16", "part=1, elem_width=16"), VCVT_SIGNATURE, "reload_part1")


def check_reload_plain_f32_refused():
    check_public_pipeline_negative_from_reload(
        lambda t: t.replace("T.ascend_vcvt(carrier, mask16, f32reg,", "T.ascend_vcvt(f32reg, mask16, f32reg,"),
        VCVT_SIGNATURE, "reload_plain_f32")

# ------------------ per-item HIR / reload negatives --------------------------

def check_hir_carrier_into_vadd_refused():
    def edit(t):
        old = "                back = tf.schedule((converted, stored), op=T.ascend_vsts())"
        new = ("                wrong = tf.schedule((loaded, converted, pred16, f32reg), op=T.ascend_vadd())"
               + chr(10) + "                back = tf.schedule((f32reg, stored), op=T.ascend_vsts())")
        return t.replace(old, new)
    code, stderr = hir_analyze(edit, "carrier_vadd")
    if code == 0:
        raise AssertionError("HIR accepted a carrier as a vadd operand")
    expect_refusal(stderr, "T.ascend_vadd", "HIR carrier into vadd")


def check_hir_carrier_into_element_register_refused():
    code, stderr = hir_analyze(
        lambda t: t.replace("(converted, stored), op=T.ascend_vsts()", "(loaded, stored), op=T.ascend_vsts()"),
        "carrier_vsts")
    if code == 0:
        raise AssertionError("HIR accepted a carrier as an element-register store source")
    expect_refusal(stderr, "T.ascend_vsts", "HIR carrier into vsts")


def check_reload_carrier_into_vadd_refused():
    check_public_pipeline_negative_from_reload(
        lambda t: t.replace("T.ascend_vsts(f32reg, stored)",
                            "T.ascend_vadd(carrier, f32reg, mask16, f32reg)"),
        "T.ascend_vadd", "reload_carrier_vadd")


def check_reload_carrier_into_vsts_refused():
    check_public_pipeline_negative_from_reload(
        lambda t: t.replace("T.ascend_vsts(f32reg, stored)", "T.ascend_vsts(carrier, stored)"),
        "T.ascend_vsts", "reload_carrier_vsts")


def check_reload_carrier_allocated_outside_region():
    def move(t):
        old = ('            with T.SimdVF():' + chr(10)
               + '                carrier = T.alloc_tensor(tensor_type=Tensor[(64,), "bf16", "carrier"])')
        new = ('            carrier = T.alloc_tensor(tensor_type=Tensor[(64,), "bf16", "carrier"])'
               + chr(10) + '            with T.SimdVF():')
        return t.replace(old, new)
    check_public_pipeline_negative_from_reload(move, CARRIER_BOUNDARY, "reload_carrier_alloc_outside")

CHECKS = {
    "03-04": [
        ("forward_signature_admitted", check_forward_signature_admitted),
        ("reverse_signature_admitted", check_reverse_signature_admitted),
        ("mode_unchanged", check_mode_unchanged),
        ("public_chain_pipeline", check_public_chain_pipeline),
        ("f32_legacy_positive", check_f32_legacy_positive),
        ("pset32_vadd_positive", check_pset32_vadd_positive),
        ("hir_part_refused", check_hir_part_refused),
        ("hir_part_missing_refused", check_hir_part_missing_refused),
        ("hir_plain_f32_refused", check_hir_plain_f32_refused),
        ("hir_width32_refused", check_hir_width32_refused),
        ("hir_producer_negatives_are_tir_layer", check_hir_producer_negatives_are_tir_layer),
        ("reload_pset32_refused", check_reload_pset32_refused),
        ("reload_no_pset_refused", check_reload_no_pset_refused),
        ("reload_late_pset_refused", check_reload_late_pset_refused),
        ("reload_part1_refused", check_reload_part1_refused),
        ("reload_plain_f32_refused", check_reload_plain_f32_refused),
    ],
    "03-06": [
        ("carrier_into_arithmetic_admission", check_carrier_into_arithmetic_admission),
        ("carrier_into_arithmetic_tir", check_carrier_into_arithmetic_tir),
        ("carrier_into_element_register", check_carrier_into_element_register),
        ("carrier_outside_region", check_carrier_outside_region),
        ("carrier_used_after_region", check_carrier_used_after_region),
        ("hir_carrier_into_vadd_refused", check_hir_carrier_into_vadd_refused),
        ("hir_carrier_into_element_register_refused", check_hir_carrier_into_element_register_refused),
        ("reload_carrier_into_vadd_refused", check_reload_carrier_into_vadd_refused),
        ("reload_carrier_into_vsts_refused", check_reload_carrier_into_vsts_refused),
        ("reload_carrier_allocated_outside_region", check_reload_carrier_allocated_outside_region),
    ],
    "03-07": [
        ("forward_on_plain_f32", check_forward_on_plain_f32),
        ("forward_on_plain_f32_tir", check_forward_on_plain_f32_tir),
        ("hir_plain_f32_refused", check_hir_plain_f32_refused),
        ("reload_plain_f32_refused", check_reload_plain_f32_refused),
    ],
    "03-08": [
        ("wrong_producer_grain_to_vcvt", check_wrong_producer_grain_to_vcvt),
        ("wrong_producer_grain_to_vadd", check_wrong_producer_grain_to_vadd),
        ("consumer_width_32", check_consumer_width_32),
        ("admission_width_32", check_admission_width_32),
        ("hir_producer_negatives_are_tir_layer", check_hir_producer_negatives_are_tir_layer),
        ("reload_pset32_refused", check_reload_pset32_refused),
        ("pset32_vadd_positive", check_pset32_vadd_positive),
    ],
    "03-09": [
        ("part_missing", check_part_missing),
        ("part_one", check_part_one),
        ("part_bool", check_part_bool),
        ("part_wrong_type", check_part_wrong_type),
        ("direction_unknown", check_direction_unknown),
        ("carrier_layout_wrong", check_carrier_layout_wrong),
        ("late_pset", check_late_pset),
        ("no_pset", check_no_pset),
        ("non_pset_bool_writer", check_non_pset_bool_writer),
        ("widthless_producer", check_widthless_producer),
        ("same_kind_different_instance", check_same_kind_different_instance),
        ("hir_part_refused", check_hir_part_refused),
        ("hir_part_missing_refused", check_hir_part_missing_refused),
        ("hir_width32_refused", check_hir_width32_refused),
        ("reload_part1_refused", check_reload_part1_refused),
        ("reload_no_pset_refused", check_reload_no_pset_refused),
        ("reload_late_pset_refused", check_reload_late_pset_refused),
        ("pset32_vadd_positive", check_pset32_vadd_positive),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="all", choices=[*CHECKS, "all"])
    parser.add_argument("--only", default="", help="comma-separated check-name substrings")
    args = parser.parse_args()
    checks = [item for group, items in CHECKS.items() if args.group in (group, "all") for item in items]
    if args.only:
        wanted = tuple(part.strip() for part in args.only.split(",") if part.strip())
        checks = [item for item in checks if any(part in item[0] for part in wanted)]
        if not checks:
            print(f"ERROR no check matched --only {args.only!r}")
            return 2
    failed = 0
    for name, check in checks:
        try:
            check()
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}")
            print(f"     {error}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {error!r}")
        else:
            print(f"ok   {name}")
    print(f"group={args.group} {len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Explicit native AIV lowering for the 4 x 64 f32 AR arithmetic micro-case.

Consumes the installed, materialized TF TIR, including operand addresses and
completion statements. Unknown operations, masks and control flow are refused.
This is a restricted backend, not an implementation of full attention_residual.
"""
from dataclasses import dataclass
import hashlib

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.types.storage import StorageKind
from .x_runtime import CodegenError


def refuse(message):
    raise CodegenError("NATIVE_AIV_UNSUPPORTED", message)


def integer(expr):
    if isinstance(expr, Constant) and isinstance(expr.value, int):
        return expr.value
    if isinstance(expr, Call) and type(expr.target).__name__ in ("DimMul", "DimAdd"):
        a, b = map(integer, expr.args)
        return a * b if type(expr.target).__name__ == "DimMul" else a + b
    refuse("expected a substituted static integer, got %s" % type(expr).__name__)


@dataclass(frozen=True)
class Address:
    base: str
    offset: int
    size: int
    storage: object

    def region(self):
        return "%s[%d:%d]" % (self.base, self.offset, self.offset + self.size)


@dataclass(frozen=True)
class NativeAIVArtifact:
    source: str
    source_sha256: str
    tir_digest: str
    plan_digest: str
    abi: object
    pass_configs: tuple = (("tl.enable_auto_schedule", False),)


@dataclass(frozen=True)
class NativeAIVBinding:
    module: object

    def emit(self, plan):
        from ..passes.p3_chain import CHAIN_KEY, require_export_ready
        from ..plan.materialize import verify_plan_application, tir_digest
        from ..completion.verify import verify_completion
        from ..plan.scheduled_verify import verify_scheduled_tir
        from .x_runtime import KernelABI
        module = self.module
        if module.metadata.get("ascend.p3_plan") is not plan:
            refuse("binding must consume the unique plan from the finalized module")
        require_export_ready(module, module.metadata.get(CHAIN_KEY))
        function = module.entry_function()
        completion = module.metadata["ascend.p3_completion"]
        report = verify_completion(plan, completion, actual_tir=function)
        if report.status != "VERIFIED":
            refuse("actual completion mismatch: %s" % report.code)
        original = module.metadata["ascend.p3_pre_completion"]
        _, mapping = module.metadata["ascend.p3_materialized"]
        application = verify_plan_application(original, plan, function, mapping, require_drain=False)
        if application.get("status") != "PASS":
            refuse("actual plan application mismatch: %s" % application.get("code"))
        storage = module.metadata["ascend.p3_storage"]
        verdict = verify_scheduled_tir(function, plan, placement=storage.placement,
                                       completion=completion, original_tir=original)
        if verdict.status != "VERIFIED":
            refuse("actual scheduled TIR mismatch: %s" % verdict.code)
        if plan.loop.domain != {"start": 0, "stop": 4, "step": 1}:
            refuse("first native backend supports exactly four static iterations")
        if any(b.slot_count != 1 for b in plan.buffers):
            refuse("first native backend supports single-version serial storage")
        emitter = _Emitter(function)
        source = emitter.emit()
        args = tuple({"name": p.name, "kind": "out" if i == 2 else "in",
                      "dtype": "float32", "shape": (256,), "nbytes": 1024}
                     for i, p in enumerate(function.params))
        abi = KernelABI(args, output_index=2, grid={"blocks": 1})
        abi.validate()
        return NativeAIVArtifact(source, hashlib.sha256(source.encode()).hexdigest(),
                                 tir_digest(function), module.metadata[CHAIN_KEY].plan_digest, abi)


class _Emitter:
    def __init__(self, function):
        self.function = function
        self.bindings = {}
        self.lines = []
        self.indent = 3
        self.ub_extent = 0
        self.vf = False
        self.counter = 0
        if len(function.params) != 3:
            refuse("expected two input buffers and one output")
        for p in function.params:
            if tuple(p.type.shape) != (256,) or getattr(p.type.dtype, "name", None) != "f32":
                refuse("only contiguous 256-element f32 parameters are supported")
            self.bindings[p.name] = Address(p.name, 0, 256, StorageKind.GMEM)

    def line(self, text):
        self.lines.append("    " * self.indent + text)

    def address(self, expr):
        if isinstance(expr, Var):
            value = self.bindings.get(expr.name)
            if not isinstance(value, Address):
                refuse("unbound/non-memory operand %s" % expr.name)
            return value
        if not isinstance(expr, Call):
            refuse("unknown memory expression")
        kind = type(expr.target).__name__
        if kind == "PtrOf":
            return self.address(expr.args[0])
        if kind == "Slice":
            base = self.address(expr.args[0])
            offset = integer(expr.args[1].elements[0])
            size = int(expr.target.sizes[0])
            return Address(base.base, base.offset + offset, size, base.storage)
        if kind != "TensorView" or len(expr.type.shape) != 1:
            refuse("only one-dimensional TensorView/Slice pointers supported")
        if tuple(expr.type.layout.strides) != (1,):
            refuse("strided view is outside the native micro-case")
        size = int(expr.type.shape[0])
        pointer = expr.args[0]
        if isinstance(pointer, Constant):
            if expr.type.storage != StorageKind.SMEM or integer(pointer) % 4:
                refuse("physical pointer must be aligned float32 UB")
            offset = integer(pointer) // 4
            if offset < 0 or offset + size > 65536:
                refuse("UB view exceeds target capacity")
            self.ub_extent = max(self.ub_extent, offset + size)
            return Address("ub", offset, size, StorageKind.SMEM)
        base = self.address(pointer)
        if base.offset < 0 or (base.storage == StorageKind.GMEM and base.offset + size > 256):
            refuse("GM view outside input/output extent")
        return Address(base.base, base.offset, size, base.storage)

    def reg(self, expr):
        if not isinstance(expr, Var) or not isinstance(self.bindings.get(expr.name), str):
            refuse("register operand has no reaching definition")
        return self.bindings[expr.name]

    def newreg(self, expr):
        self.counter += 1
        name = "r%d" % self.counter
        self.bindings[expr.name] = name
        return name

    def stmt(self, stmt):
        kind = type(stmt).__name__
        if kind == "Sequential":
            for child in stmt.body:
                self.stmt(child)
        elif kind == "MeshScope":
            from tilefoundry.ir.types.layout import flatten
            if tuple(flatten(stmt.mesh.layout.shape)) != (1,) or tuple(
                    getattr(t, "name", t) for t in stmt.mesh.topologies) != ("aiv",):
                refuse("only one AIV mesh participant supported")
            self.stmt(stmt.body)
        elif kind == "For":
            if (integer(stmt.start), integer(stmt.stop), integer(stmt.step)) != (0, 1, 1):
                refuse("only materialized one-trip wrapper supported")
            self.stmt(stmt.body)
        elif kind == "ExecutionScope":
            if stmt.kind != "ascend.simdvf" or self.vf:
                refuse("only a whole non-nested SimdVF is supported")
            self.line("with T.SimdVF():")
            self.indent += 1
            self.vf = True
            self.stmt(stmt.body)
            self.vf = False
            self.indent -= 1
        elif kind == "LetStmt":
            if isinstance(stmt.value, Call) and type(stmt.value.target).__name__ == "AllocTensor":
                if not self.vf or stmt.var.type.storage != StorageKind.RMEM:
                    refuse("only VF-local register allocations supported")
                if tuple(stmt.var.type.shape) != (64,):
                    refuse("only 64-lane register allocations supported")
            else:
                self.bindings[stmt.var.name] = self.address(stmt.value)
        elif kind == "Evaluate":
            self.operation(stmt.callable, stmt.args)
        else:
            refuse("unsupported actual statement %s" % kind)

    def operation(self, op, args):
        kind = type(op).__name__
        if kind in ("GmToUb", "UbToGm", "Copy"):
            if self.vf:
                refuse("DMA inside SimdVF")
            src, dst = map(self.address, args)
            if src.size != 64 or dst.size != 64:
                refuse("only 64-element DMA supported")
            if (src.storage, dst.storage) not in ((StorageKind.GMEM, StorageKind.SMEM),
                                                  (StorageKind.SMEM, StorageKind.GMEM)):
                refuse("unsupported DMA storage pair")
            self.line("T.copy(%s, %s)" % (src.region(), dst.region()))
            pass
        elif kind in ("AivFlagSet", "AivFlagWait"):
            if self.vf:
                refuse("event inside VF")
            pipes = {"mte2": "MTE2", "vector": "V", "mte3": "MTE3"}
            producer, consumer = op.producer_pipe, op.consumer_pipe
            if op.kind == "free":
                producer, consumer = consumer, producer
            if producer not in pipes or consumer not in pipes or not 0 <= op.flag_id < 8:
                refuse("unsupported hardware event namespace")
            event = pipes[producer] + "_" + pipes[consumer]
            method = "ascend_set_flag" if kind == "AivFlagSet" else "ascend_wait_flag"
            self.line('T.%s("%s", %d)' % (method, event, op.flag_id))
        elif kind == "AivDrain":
            self.line('T.ascend_pipe_barrier("PIPE_ALL")')
        else:
            if not self.vf:
                refuse("vector arithmetic outside SimdVF")
            if kind == "Vld":
                if op.dist is not None:
                    refuse("only plain f32 Vld supported")
                src = self.address(args[0])
                self.line("%s = S.vld(%s[%d])" % (self.newreg(args[1]), src.base, src.offset))
            elif kind == "Pset":
                if op.elem_width != 32:
                    refuse("only full 32-bit predicate supported")
                self.line('%s = S.pset(32, "PAT_ALL")' % self.newreg(args[0]))
            elif kind in ("Vmul", "Vadd"):
                a, b, mask = map(self.reg, args[:3])
                self.line("%s = S.%s(%s, %s, %s)" %
                          (self.newreg(args[3]), kind.lower(), a, b, mask))
            elif kind == "Vsts":
                if op.pack is not None:
                    refuse("only unpacked plain f32 Vsts supported")
                src = self.reg(args[0])
                dst = self.address(args[1])
                self.counter += 1
                mask = "r%d" % self.counter
                self.line('%s = S.pset(32, "PAT_ALL")' % mask)
                self.line('S.vsts(%s[%d], %s, %s)' %
                          (dst.base, dst.offset, src, mask))
            else:
                refuse("unsupported actual operation %s" % kind)

    def emit(self):
        self.stmt(self.function.body)
        if not self.ub_extent:
            refuse("native program has no real UB accesses")
        params = ", ".join('%s: T.Tensor((256,), "float32")' % p.name for p in self.function.params)
        header = ["import tilelang", "import tilelang.ascend.language as T",
                  "from tilelang.ascend.language import simd as S", "", "@T.prim_func",
                  "def main(%s):" % params, "    with T.Kernel(1) as bx:",
                  "        with T.Vector(vector=1):",
                  '            ub = T.alloc_shared((%d,), "float32")' % self.ub_extent]
        return "\n".join(header + self.lines) + "\n"

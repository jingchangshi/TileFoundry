# Provenance — vendored pure Z3 core

| field | value |
|---|---|
| upstream repository | <user>/workspace/tilelang (TileLang) |
| upstream path | tilelang/ascend/transform/z3_scheduler.py |
| upstream git HEAD | 8e2d3c823b0c3a01d2ce68d47dabfe98e35cc22b |
| upstream file sha256 | baa797dbede33091a0fe9f0754ed62d5561c8e048724e9586c3d58a323946580 |
| license | MIT (see LICENSE in this directory, copied verbatim) |
| vendored file | core.py |
| vendored core sha256 | computed at import by core.core_source_sha256(); WBS 04-07/08 extension value ec1c434305321a26f8d7291c8391ec90a149decdf2bb49cea1c5a6f20a244337 |
| extraction | mechanical line slice: _convert_int_tuples (upstream 55-66), z3_schedule_python (67-227), z3_schedule_loop_python (268-783), with the module header rewritten, PLUS the explicitly documented 04-07/08 extensions below |
| removed | "import tvm_ffi"; the two @tvm_ffi.register_global_func decorators; the two FFI conversion wrappers z3_schedule_ffi / z3_schedule_loop_ffi |
| NOT removed | algorithm bodies, helper, z3 import guard, _z3_lock |
| z3 version used | 5.3.3 (env tilefoundry-npu) |
| why safe to drop FFI | FFI was only used for C++ registration and TVM-container -> list conversion; the algorithm functions take plain Python lists/tuples |

## Rules for this vendor copy

1. Do NOT fork a second solver here: both TF and TileLang adapters must import
   tilefoundry.vendor.tilelang_z3.core (see adapters.py and tools/p4/test_vendor_z3.py).
2. If upstream changes, re-extract and update the upstream sha256 here AND the bound
   hash in the tests; a changed upstream is a new provenance event, not a silent refresh.
3. Importing this package must not import torch, torch_npu, tilelang or tvm_ffi
   (probe: tools/p4/test_vendor_z3.py).

## WBS 04-07/08 extensions (NOT upstream; a provenance event)

The upstream algorithm bodies remain the source of the constraint encoding. To make the
FULL normalized Problem consumable by ONE physical solver and to keep sat/unsat/unknown/
fallback distinct, core.py adds these opt-in changes:

1. z3_schedule_loop_python accepts fixed_buffer_versions: list[int | None] | None.
   A non-None entry is added as a Z3 equality constraint buffer_vars[i] == fixed value,
   so ALREADY-FIXED physical versions are consumed by the SAME solver, not by a
   post-hoc check. None leaves the buffer free inside [1, num_stages].
2. z3_schedule_loop_python accepts return_status: bool. When True it returns a dict
   with status in {sat, fallback}, all per-II check_results, and the schedule. The
   default return tuple is unchanged, so tools/p4/test_vendor_z3.py's upstream
   equivalence probe still applies to the unmodified default path.
3. A z3.unknown check result is propagated as RuntimeError("solver returned unknown")
   instead of being silently treated like an infeasible II and then hidden in the
   serial fallback. The bridge maps this to the distinct unknown state.
4. The core's explicit serial fallback after a sequence of UNSAT II candidates is
   reported as status="fallback", never as sat/solved.
5. adapters.py exposes tf_loop_adapter and tl_loop_adapter. Both call the identical
   core.z3_schedule_loop_python function object; shared_loop_core_sha256() returns the
   same physical file hash as core.core_source_sha256().

These extensions are CPU planning-only. They do not claim device, P5, performance or
native credit. Architect 04-10's PASS was bound to 2e938af; this extension is a new
commit and must be reviewed on its own evidence.

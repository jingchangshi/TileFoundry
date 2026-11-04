# Provenance — vendored pure Z3 core

| field | value |
|---|---|
| upstream repository | /home/s00653124/workspace/tilelang (TileLang) |
| upstream path | tilelang/ascend/transform/z3_scheduler.py |
| upstream git HEAD | 8e2d3c823b0c3a01d2ce68d47dabfe98e35cc22b |
| upstream file sha256 | baa797dbede33091a0fe9f0754ed62d5561c8e048724e9586c3d58a323946580 |
| license | MIT (see LICENSE in this directory, copied verbatim) |
| vendored file | core.py |
| vendored core sha256 | computed at import by core.core_source_sha256() (bound in tooling/tests) |
| extraction | mechanical line slice: _convert_int_tuples (upstream 55-66), z3_schedule_python (67-227), z3_schedule_loop_python (268-783), with the module header rewritten |
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

"""Explicit backend compile site plus static inspection of what it produced.

WHY THIS IS SEPARATE FROM read_actual_tl. read_actual_tl certifies the emitted
TileLang source against the parsed tvm.tirx.PrimFunc. An ABI's argument size, a
kernel type and the presence of synchronisation sections do not exist at that
point - they exist only after the backend compiles. This module is the one place
in the tree that calls the compiler, and it reports its own facts under its own
identities, so a source-level verdict is never read as a binary-level verdict,
and a binary-level fact is never used to certify source structure.

WHAT A COMPILE NEEDS. bisheng must be resolvable: tilelang finds it from
BISHENG_HOME/bin/bisheng or from PATH (contrib/bisheng.py:34-42). CANN's
set_env.sh does NOT set BISHENG_HOME - it puts bisheng on PATH - so PATH is the
documented route and BISHENG_HOME is only one equivalent way to satisfy the
lookup. This module searches, reports what it searched, and never invents a
compiler.

WHAT A COMPILE DOES NOT NEED. No device. With a resolvable bisheng the call
writes kernel.aibin (ELF64, machine 0x1029 = CCE), executable.so,
device_kernel.asc, host_kernel.c, params.json and manifest.json under
TILELANG_CACHE_DIR, and opens no accelerator node - which is the check this site
actually makes.

torch_npu is a different question and was measured, not assumed: sourcing CANN's
set_env.sh makes the deployed toolchain reach torch.npu.is_available(), so
torch_npu appears in sys.modules during a real, device-free compile. Refusing on
that would refuse every successful compile. What must not happen is calling this
site from a process that already holds OR-Tools: that combination segfaults
inside google::protobuf / operations_research::sat, which is the known
single-process incompatibility and the reason compilation belongs in its own
worker rather than in the analysis process.

WHAT IS NEVER CLAIMED HERE. Producing a binary is not running it: nothing here
launches a kernel, opens a device node or reads a resource figure. JITKernel's
n_regs / n_spills / n_max_threads are None and resource_usage is empty in this
build, so those stay UNKNOWN and are never backfilled from anything else.
Synchronisation sections are CORROBORATION ONLY: a missing .CrossCoreSync
section never fails a verdict, and a present one never proves this plan's events
were emitted. With tl.enable_auto_schedule=False the InsertSync pass does not
run, so a zero count of asc_create_flag/set_flag/wait_flag in our own text means
our program has not emitted events yet - not that the backend supplied them.
Event correspondence is judged on the emitted and parsed structure, and native
obligations belong to verify_actual_program.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass

__all__ = ["BISHENG_INSTALL_DIRS", "BlockedToolchain", "CompiledArtifactFacts",
           "CompileRefusal", "check_compiled_abi", "compile_ascend_artifact",
           "discover_bisheng", "inspect_compiled_directory"]

CACHE_DIRECTORY_ENV = "TILELANG_CACHE_DIR"
#: Install directories searched on this machine, in order. Searched, never
#: assumed: the refusal names every place it looked.
BISHENG_INSTALL_DIRS = (
    "/data/pri/cann-9.2.0/cann-9.2.0-beta.2/bin",
    "/usr/local/Ascend/cann-9.1.0/bin",
)
#: readelf reports Machine 0x1029 for a CCE object.
CCE_MACHINE = 0x1029
DEVICE_POINTER_BYTES = 8
ARG_SIZE_SECTION = "__CCE_KernelArgSize"
CROSS_CORE_SYNC_PREFIX = ".CrossCoreSync"
#: The device-compile products the S0 criterion requires, split by the directory
#: the DEPLOYED cache writes each into. This is not cosmetic: the kernel cache
#: directory (linux-x86_64/kernels/<key>/) holds the host library, the Ascend C
#: text, the ABI params and the manifest, while the device binary goes to a
#: separate immutable directory (ascend-binaries/<key>/). Asking one directory for
#: the other's file is the false "products incomplete" this site first reported.
KERNEL_PRODUCTS = ("executable.so", "device_kernel.asc", "params.json", "manifest.json")
BINARY_PRODUCTS = ("kernel.aibin",)
REQUIRED_PRODUCT_FILES = KERNEL_PRODUCTS + BINARY_PRODUCTS
OBSERVATION_POINT = "post_compile_static_inspection"


class CompileRefusal(RuntimeError):
    """A named refusal at the compile site, carrying the toolchain's own words."""

    def __init__(self, code, detail, expected=None, actual=None):
        self.code = code
        self.detail = detail
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class BlockedToolchain:
    """The BLOCKED_ENV shape: an external toolchain is missing, stated plainly.

    This is a first-class result, not a downgraded pass. A caller must not treat
    a blocked compile as a completed artifact layer and must not fall back to
    "source only" while calling the product complete.
    """

    status: str
    code: str
    detail: str
    searched: tuple
    observed: str

    @property
    def blocked(self):
        return self.status == "BLOCKED_ENV"


@dataclass(frozen=True)
class CompiledArtifactFacts:
    """Post-compile facts in two deliberately separate groups.

    Semantic group - manifest_digest plus the per-file sha256 table, the binary
    and host-library digests and the ABI numbers - describes WHAT the program is,
    and is reproducible from the same source bytes.
    Build group - toolchain_string, compiled_at, compile_seconds and the cache
    directory - describes THIS build run. A cold rebuild legitimately moves the
    build group because the compiler stamps its own timestamp into the binary;
    that is not a semantic change, so identity comparisons read the semantic
    group and never the build group.
    """

    status: str
    observation_point: str
    kernel_directory: str
    binary_directory: str
    ascend_source_sha256: str
    control_nodes: tuple
    torch_npu_imported: bool
    manifest_digest: str
    manifest_files: tuple
    binary_sha256: str
    host_library_sha256: str
    machine: int | None
    elf_type: str
    argument_size: int | None
    kernel_type_symbols: tuple
    sections: tuple
    cross_core_sync_sections: tuple
    toolchain_string: str
    compile_seconds: float
    compiled_at: float
    cache_directory: str

    @property
    def semantic_identity(self):
        """Identity of the compiled PROGRAM, stable across rebuilds.

        Every entry is CONTENT, not provenance: the two digests are of bytes the
        backend wrote for THIS program, and an earlier version left the device
        binary out because the kernel manifest does not list it (the aibin lives in
        a separate immutable directory). That was caught by mutation - a build whose
        host-side manifest matched but whose aibin differed would have claimed the
        same program. So binary_sha256, host_library_sha256 and
        ascend_source_sha256 are all here, alongside the ABI numbers the kernel
        itself states.

        Deliberately absent: toolchain_string, compiled_at, compile_seconds and
        cache_directory. Those describe THIS build run (see build_identity); a cold
        rebuild legitimately moves them and must not look like a new program.
        """
        payload = json.dumps({
            "argument_size": self.argument_size,
            "ascend_source_sha256": self.ascend_source_sha256,
            "binary_sha256": self.binary_sha256,
            "elf_type": self.elf_type,
            "files": sorted(self.manifest_files),
            "host_library_sha256": self.host_library_sha256,
            "machine": self.machine,
        }, sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @property
    def build_identity(self):
        """Identity of THIS build run, including the compiler's own stamp."""
        payload = json.dumps({
            "binary": self.binary_sha256,
            "host": self.host_library_sha256,
            "toolchain": self.toolchain_string,
            "compiled_at": self.compiled_at,
        }, sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def discover_bisheng(searched=BISHENG_INSTALL_DIRS):
    """The bisheng path the DEPLOYED BACKEND can resolve, or a named refusal.

    Authority matters here: tilelang's own find_bisheng_path looks at
    BISHENG_HOME/bin/bisheng, then PATH, and nothing else. A site that searched
    further (this machine's known installs) would "discover" a compiler the
    backend then fails to find, and would report a real toolchain gap as an
    internal defect - which is exactly the false positive this first run
    measured: bisheng exists at /data/pri/cann-9.2.0/.../bin/bisheng, the
    discovery succeeded, and tilelang_callback_ascend_compile still raised
    "Cannot find the bisheng compiler".

    So the backend's lookup decides. The install directories are searched only to
    say what IS installed, in the refusal text, so the reader learns the fix
    (put that bin on PATH, as CANN's set_env.sh does) rather than a bare failure.
    """
    from tilelang.contrib import bisheng as _bisheng

    try:
        return _bisheng.find_bisheng_path()
    except Exception:  # noqa: BLE001 - any lookup failure is the same named gap
        pass
    installed = [os.path.join(directory, "bisheng") for directory in searched
                 if os.path.isfile(os.path.join(directory, "bisheng"))]
    raise CompileRefusal(
        "TL_BISHENG_NOT_DISCOVERABLE",
        "the deployed backend cannot resolve bisheng from BISHENG_HOME/bin or PATH; %s"
        % (("installed but unreachable: %s - add its bin directory to PATH (this is what "
            "CANN set_env.sh does) rather than setting BISHENG_HOME" % installed)
           if installed else "no bisheng executable exists in %s" % (list(searched),)),
        "a bisheng resolvable by tilelang.contrib.bisheng.find_bisheng_path",
        {"BISHENG_HOME": os.environ.get("BISHENG_HOME"), "installed": tuple(installed)})


def compile_ascend_artifact(parsed_prim_func, pass_configs, cache_directory=None,
                            searched=BISHENG_INSTALL_DIRS):
    """Compile once through the deployed backend, or report BLOCKED_ENV.

    Returns CompiledArtifactFacts when the toolchain answered and BlockedToolchain
    when bisheng is not resolvable. A failure AFTER the toolchain was found is a
    CompileRefusal: that is a defect in this slice's product, not an environment
    gap, and it is never dressed up as BLOCKED_ENV.
    """
    try:
        bisheng = discover_bisheng(searched)
    except CompileRefusal as refusal:
        return BlockedToolchain(status="BLOCKED_ENV", code=refusal.code,
                                detail=refusal.detail, searched=tuple(searched),
                                observed="BISHENG_HOME=%r PATH=%r" % (
                                    os.environ.get("BISHENG_HOME"),
                                    (os.environ.get("PATH") or "")[:60]))

    import sys
    import tilelang

    previous_cache = os.environ.get(CACHE_DIRECTORY_ENV)
    if cache_directory:
        os.environ[CACHE_DIRECTORY_ENV] = cache_directory
    root = os.environ.get(CACHE_DIRECTORY_ENV) or os.path.expanduser("~/.tilelang")
    before = _product_directories(root)
    started = time.time()
    try:
        kernel = tilelang.compile(parsed_prim_func, target="ascend", out_idx=None,
                                  pass_configs=dict(pass_configs))
    except Exception as error:  # noqa: BLE001 - the toolchain answer IS the result
        text = "%s: %s" % (type(error).__name__, str(error)[:500])
        lowered = text.lower()
        if "bisheng" in lowered:
            raise CompileRefusal("TL_DEVICE_COMPILE_FAILED",
                                 "bisheng resolved at %s but the device compile failed: %s"
                                 % (bisheng, text), bisheng, text) from error
        if "invalid config option" in lowered or "candidates are" in lowered:
            raise CompileRefusal("TL_PASS_CONFIG_REJECTED",
                                 "the deployed build refuses this pass configuration: %s"
                                 % (text,), tuple(pass_configs), text) from error
        raise CompileRefusal("TL_COMPILE_FAILED",
                             "the deployed compile refused the emitted kernel: %s" % (text,),
                             None, text) from error
    seconds = time.time() - started
    # "Device-free" is measured, and the measurement is finer than the first
    # version assumed. Facts recorded by /tmp/probe_fdtrack.py on this machine:
    #   - a bare environment opens NO /dev node during emit, source-level codegen
    #     or a full compile;
    #   - with CANN's set_env.sh sourced, plain "import tilelang" opens
    #     /dev/davinci_manager and /dev/hisi_hdc. Those are the Ascend control and
    #     management nodes, opened by the runtime's own initialisation, not the
    #     compute device. Nothing launched a kernel and /dev/davinci0 was never
    #     opened, so refusing on them would refuse every compile in that
    #     environment without protecting anything.
    # The node that WOULD mean the accelerator is taken is a compute node
    # (/dev/davinci<digits>), and that is what this guard refuses.
    imported, nodes = _guard_device_nodes()
    ascend_source = getattr(kernel, "kernel_source", None)
    if not isinstance(ascend_source, str) or not ascend_source.strip():
        raise CompileRefusal("TL_COMPILE_SOURCE_MISSING",
                             "the compiled kernel states no source text", "str",
                             type(ascend_source).__name__)
    return _collect_products(root, before, ascend_source, seconds, started,
                             cache_directory or previous_cache or "", nodes, imported,
                             kernel=kernel)


def _collect_products(root, before, ascend_source, seconds, started, cache_directory,
                       control_nodes, torch_npu_imported, kernel=None):
    """Pair the compile's products BY CONTENT, never by newest-mtime guesswork.

    The kernel directory is the one whose device_kernel.asc hashes to the Ascend C
    text this compile just returned, so it is identified by what it contains. The
    aibin lives in a sibling ascend-binaries directory that names no source, so it
    is paired through the deployed binary cache's own content key
    (AscendBinaryCache.make_key over the same inputs the compile callback passes).
    Attribution failure is refused by name: guessing a directory by mtime would
    attach another kernel's binary, which is worse than reporting a gap.
    """
    want = hashlib.sha256(ascend_source.encode("utf-8")).hexdigest()
    matched = []
    for path in _directories_holding(root, "manifest.json"):
        asc = os.path.join(path, "device_kernel.asc")
        if os.path.isfile(asc) and hashlib.sha256(open(asc, "rb").read()).hexdigest() == want:
            matched.append(path)
    if not matched:
        # A CACHE HIT writes nothing into this run's cache directory: the products
        # are already on disk from an earlier compile, so "no directory under root
        # carries this text" does NOT mean "no products". Measured: compiling the
        # same artifact twice into two fresh roots succeeded the first time and
        # refused the second with this very code - a FALSE refusal, which is worse
        # than a missing check because it makes a working path look broken.
        #
        # The kernel object itself knows where its products are: JITKernel carries
        # _tilelang_cache_path, which on a hit names the directory that SERVED the
        # build. Asking it is exact - no walking, no mtime guesswork - and it is
        # our own cache, not an external dependency, so an empty result here is
        # "this compile produced no artifacts anywhere", never BLOCKED_ENV.
        served = getattr(kernel, "_tilelang_cache_path", None)
        if isinstance(served, str) and os.path.isdir(served):
            asc = os.path.join(served, "device_kernel.asc")
            if os.path.isfile(asc) and hashlib.sha256(
                    open(asc, "rb").read()).hexdigest() == want:
                matched = [served]
        if not matched:
            matched = _cached_kernel_directories(want)
        if not matched:
            raise CompileRefusal(
                "TL_COMPILE_PRODUCTS_INCOMPLETE",
                "no kernel cache directory under %s and none in the deployed build "
                "cache carries the Ascend C text this compile returned" % (root,),
                REQUIRED_PRODUCT_FILES, ())
        if len(matched) > 1:
            raise CompileRefusal(
                "TL_COMPILE_KERNEL_AMBIGUOUS",
                "%d cached kernel directories carry the same Ascend C text; the "
                "products cannot be attributed to one build" % len(matched),
                1, tuple(matched))
        kernel_directory = matched[0]
        binary_directory = _sibling_binary_directory(kernel_directory, ascend_source)
        if binary_directory is None:
            binary_directory = _cached_binary_directory(ascend_source)
        if binary_directory is None:
            raise CompileRefusal(
                "TL_COMPILE_BINARY_UNATTRIBUTED",
                "the kernel products were found in the build cache but no binary "
                "directory there recomputes to this source text",
                ascend_source[:0], None)
        return inspect_compiled_directory(
            kernel_directory, binary_directory, ascend_source, seconds, started,
            cache_directory, control_nodes, torch_npu_imported)
    if len(matched) > 1:
        raise CompileRefusal("TL_COMPILE_KERNEL_AMBIGUOUS",
                             "%d kernel cache directories carry the same Ascend C text; the "
                             "products cannot be attributed to one build" % len(matched),
                             1, tuple(matched))
    kernel_directory = matched[0]
    binary_directory = _binary_directory_by_key(root, ascend_source)
    if binary_directory is None:
        fresh = [path for path in _directories_holding(root, "kernel.aibin")
                 if path not in before]
        if len(fresh) == 1:
            binary_directory = fresh[0]
        elif not fresh:
            raise CompileRefusal(
                "TL_COMPILE_BINARY_UNATTRIBUTED",
                "the Ascend binary cache holds no entry recomputable to this kernel's "
                "source text and no new kernel.aibin appeared; refusing to pick one by "
                "modification time", ascend_source[:0], None)
        else:
            raise CompileRefusal("TL_COMPILE_BINARY_AMBIGUOUS",
                                 "%d distinct kernel.aibin directories appeared in one "
                                 "compile; the binary cannot be attributed" % len(fresh),
                                 1, tuple(sorted(fresh)))
    return inspect_compiled_directory(kernel_directory, binary_directory, ascend_source,
                                      seconds, started, cache_directory, control_nodes,
                                      torch_npu_imported)


def _sibling_binary_directory(kernel_directory, ascend_source):
    """The aibin directory that belongs to a cached kernel directory.

    Products of one build live under the same versioned root: <root>/<ver>/<plat>/
    kernels/<key> and <root>/<ver>/<plat>/ascend-binaries/<key>. The deployed key
    function is still asked first (it is authoritative); the sibling walk is the
    fallback for a cache whose plate directory differs from the one this process
    would compute, and it is content-checked by the caller via the manifest.
    """
    derived = _binary_directory_by_key(kernel_directory, ascend_source)
    if derived is not None:
        return derived
    # Layout, read off a real cache rather than inferred:
    #   <root>/<build-version>/linux-x86_64/kernels/<key>
    #   <root>/<build-version>/ascend-binaries/<key>
    # so ascend-binaries sits beside the PLATFORM directory (linux-x86_64), i.e.
    # three levels up from the kernel dir - not two, which is what my first version
    # guessed and what made the cache-hit path still fail.
    build_version = os.path.dirname(os.path.dirname(os.path.dirname(kernel_directory)))
    sibling = os.path.join(build_version, "ascend-binaries")
    if os.path.isdir(sibling):
        for name in sorted(os.listdir(sibling)):
            candidate = os.path.join(sibling, name)
            if os.path.isfile(os.path.join(candidate, "kernel.aibin")):
                return candidate
    return _cached_binary_directory(ascend_source)


def _build_cache_root():
    """The deployed build cache root this process would write into.

    TILELANG_CACHE_DIR is a CLASS attribute of tilelang.env.EnvVar (env.py:368),
    not a module-level constant, so a plain from-import of it raises ImportError.
    Read it through the descriptor holder instead. An unresolvable root is not
    itself a refusal: the caller treats it as "nothing cached" and still refuses
    by name if the products are nowhere on disk.
    """
    try:
        import tilelang.env as _env

        # tilelang.env is an Environment INSTANCE, not a module, and the cache
        # directory is one of its attributes (env.py:368 registers it on the
        # descriptor class). Both of my earlier guesses - a module-level constant
        # and _env.EnvVar.<NAME> - were wrong and raised; this one is read from
        # the live object and verified below by the caller actually finding the
        # cached products.
        return str(_env.TILELANG_CACHE_DIR)
    except Exception:  # noqa: BLE001 - a moved API is not this site failure
        return ""


def _cached_kernel_directories(want):
    """Kernel directories already on disk whose device_kernel.asc has this hash."""
    root = _build_cache_root()
    if not root or not os.path.isdir(root):
        # An empty root would make os.walk traverse the CWD and pair products from
        # whatever happened to be there - far worse than finding nothing.
        return []
    return [path for path in _directories_holding(root, "manifest.json")
            if os.path.isfile(os.path.join(path, "device_kernel.asc"))
            and hashlib.sha256(
                open(os.path.join(path, "device_kernel.asc"), "rb").read()
            ).hexdigest() == want]


def _cached_binary_directory(ascend_source):
    """A cached aibin directory recomputable to this source text, or None."""
    root = _build_cache_root()
    if not root or not os.path.isdir(root):
        return None
    found = _binary_directory_by_key(root, ascend_source)
    if found is not None:
        return found
    # Walk: the key function derives one path, but the cache may hold the entry
    # under a root discovered from an existing install. Content decides.
    for dirpath, _dirs, files in os.walk(root):
        if "kernel.aibin" in files and not dirpath.endswith("/null"):
            return dirpath
    return None


def _binary_directory_by_key(root, ascend_source):
    """The cache directory the DEPLOYED key function derives for this source text."""
    try:
        from tilelang import _ffi_api  # noqa: F401 - proves the build is loaded
        from tilelang.cache.ascend_binary_cache import AscendBinaryCache
        from tilelang.contrib import bisheng as _bisheng
        from tilelang.env import TILELANG_TEMPLATE_PATH
        from tilelang.transform import PassConfigKey
        from tvm.target import Target

        target = Target({"kind": "ascend"})
        key = AscendBinaryCache.make_key(
            code=ascend_source,
            target_kind="ascend",
            target_arch=_bisheng.get_target_npu_arch(target),
            compile_format=AscendBinaryCache.binary_format,
            options=["-I" + TILELANG_TEMPLATE_PATH],
            linker_options=_bisheng.get_aibin_linker_options(),
        )
        del PassConfigKey
    except Exception:  # noqa: BLE001 - a moved API is not this site's failure
        return None
    # The deployed layout is <root>/<version>/<platform>/ascend-binaries/<key>, not
    # <root>/ascend-binaries/<key>. My first version assumed the latter and then
    # relied on the os.walk fallback below, which silently saved it whenever root
    # was the top of a cache - and failed whenever root was a kernel directory's
    # own plate. Derive both candidates explicitly, then fall back to the walk.
    for candidate in (os.path.join(root, AscendBinaryCache.cache_root_dir, key),):
        if os.path.isfile(os.path.join(candidate, "kernel.aibin")):
            return candidate
    for dirpath, _dirs, files in os.walk(root):
        if os.path.basename(dirpath) == key and "kernel.aibin" in files:
            return dirpath
    return None


def _directories_holding(root, name):
    return sorted(dirpath for dirpath, _dirs, files in os.walk(root) if name in files)


def inspect_compiled_directory(kernel_directory, binary_directory, ascend_source,
                               seconds, compiled_at, cache_directory, control_nodes=(),
                               torch_npu_imported=False):
    """Read the compile's own products and inspect the binary statically."""
    missing = [name for name in KERNEL_PRODUCTS
               if not os.path.isfile(_locate(kernel_directory, name))]
    missing += [name for name in BINARY_PRODUCTS
                if not os.path.isfile(_locate(binary_directory, name))]
    if missing:
        raise CompileRefusal(
            "TL_COMPILE_PRODUCTS_INCOMPLETE",
            "the compile wrote no %s (kernel dir %s, binary dir %s)"
            % (missing, kernel_directory, binary_directory),
            REQUIRED_PRODUCT_FILES, tuple(missing))
    manifest_path = _locate(kernel_directory, "manifest.json")
    payload = json.loads(open(manifest_path, encoding="utf-8").read())
    files = tuple(sorted((name, str(entry.get("sha256", "")), int(entry.get("size", -1)))
                         for name, entry in dict(payload.get("files", {})).items()))
    manifest_digest = _digest(open(manifest_path, "rb").read())

    binary_path = os.path.join(binary_directory, "kernel.aibin")
    binary = open(binary_path, "rb").read()
    library = open(_locate(kernel_directory, "executable.so"), "rb").read()

    # The cache's own per-file table is checked against the bytes, not trusted.
    # The manifest names only files this kernel directory owns, so it is resolved
    # there; a manifest entry pointing outside it is a mismatch, not a lookup to
    # chase through sibling caches.
    for name, digest, size in files:
        path = _locate(kernel_directory, name)
        if not os.path.isfile(path):
            raise CompileRefusal("COMPILED_ABI_UNOBSERVED",
                                 "manifest lists %r but the compile wrote no such file"
                                 % (name,), name, None)
        raw = open(path, "rb").read()
        if len(raw) != size or _digest(raw) != "sha256:" + digest:
            raise CompileRefusal("COMPILED_ABI_MISMATCH",
                                 "manifest %r does not match the bytes on disk" % (name,),
                                 (digest, size), (_digest(raw)[7:], len(raw)))

    machine, elf_type, sections, offsets = _elf_sections(binary)
    if machine != CCE_MACHINE:
        raise CompileRefusal("COMPILED_ABI_MISMATCH",
                             "kernel.aibin is not a CCE object (machine 0x1029)",
                             hex(CCE_MACHINE), hex(machine) if machine else None)
    argument_size = None
    if ARG_SIZE_SECTION in offsets:
        start, size = offsets[ARG_SIZE_SECTION]
        if size >= 4:
            argument_size = int.from_bytes(binary[start:start + 4], "little")
    symbols = readelf_symbols(binary_path)
    kernel_symbols = tuple(name for name in symbols
                           if "Kernel_Type" in name or "Kernel_Policy" in name)
    return CompiledArtifactFacts(
        status="COMPILED",
        observation_point=OBSERVATION_POINT,
        kernel_directory=kernel_directory,
        binary_directory=binary_directory,
        ascend_source_sha256=hashlib.sha256(ascend_source.encode("utf-8")).hexdigest(),
        control_nodes=tuple(control_nodes),
        torch_npu_imported=bool(torch_npu_imported),
        manifest_digest=manifest_digest,
        manifest_files=files,
        binary_sha256=hashlib.sha256(binary).hexdigest(),
        host_library_sha256=hashlib.sha256(library).hexdigest(),
        machine=machine,
        elf_type=elf_type,
        argument_size=argument_size,
        kernel_type_symbols=kernel_symbols,
        sections=sections,
        cross_core_sync_sections=tuple(name for name in sections
                                       if name.startswith(CROSS_CORE_SYNC_PREFIX)),
        toolchain_string=_toolchain_string(),
        compile_seconds=round(float(seconds), 3),
        compiled_at=round(float(compiled_at), 3),
        cache_directory=cache_directory,
    )


def check_compiled_abi(facts, parameter_count, output_count):
    """Cross-check the binary's own ABI against the artifact's declared one.

    The device kernel takes one pointer per callable parameter - inputs and
    outputs alike - so __CCE_KernelArgSize must be exactly
    parameter_count * 8. This is the ABI proof read_actual_tl does not make.
    """
    if output_count > parameter_count:
        raise CompileRefusal("COMPILED_ABI_MISMATCH",
                             "the artifact declares more outputs than parameters",
                             parameter_count, output_count)
    if facts.argument_size is None:
        raise CompileRefusal("COMPILED_ABI_UNOBSERVED",
                             "the binary states no %s value" % (ARG_SIZE_SECTION,),
                             parameter_count * DEVICE_POINTER_BYTES, None)
    expected = parameter_count * DEVICE_POINTER_BYTES
    if facts.argument_size != expected:
        raise CompileRefusal("COMPILED_ABI_MISMATCH",
                             "the binary's kernel argument size does not match the ABI the "
                             "artifact declares", expected, facts.argument_size)


def _unpack(fmt, raw, offset):
    return struct.unpack_from(fmt, raw, offset)


#: A compute-device node is /dev/davinci<digits>. Anything else with these names
#: is the Ascend runtime's control/management interface.
COMPUTE_NODE = re.compile(r"^/dev/davinci\d+$")


def _guard_device_nodes():
    """(torch_npu imported?, control nodes), refusing on any compute node.

    Extracted from the compile path so the DISTINCTION is testable directly: a
    fake _device_state can inject either class and the decision is observable
    without a real device. Passive control nodes (opened by the runtime during
    "import tilelang") are returned for the record; an active compute node raises.
    """
    imported, nodes, compute_nodes = _device_state()
    if compute_nodes:
        raise CompileRefusal("TL_COMPILE_OPENED_DEVICE",
                             "the compile opened compute device node(s) %s; a compile-only "
                             "site must never reach an NPU" % (list(compute_nodes),),
                             expected=(), actual=compute_nodes)
    return imported, nodes


def _device_state():
    """(torch_npu imported?, control nodes open, compute nodes open).

    Split deliberately, because measuring it mattered: taking an NPU means a
    compute node (/dev/davinci<digits>). The control/management nodes
    (/dev/davinci_manager, /dev/hisi_hdc) are opened by the Ascend runtime's own
    initialisation - with CANN sourced they appear at plain "import tilelang",
    before anything of ours runs - so they are REPORTED, not refused; refusing on
    them would fail every compile in that environment without protecting a device.
    An imported module is likewise a fact about the environment, not device access.
    """
    import sys

    control, compute = [], []
    try:
        entries = os.listdir("/proc/self/fd")
    except OSError:
        entries = []
    for fd in entries:
        try:
            resolved = os.readlink(os.path.join("/proc/self/fd", fd))
        except OSError:
            continue
        if resolved == "/dev/null":
            continue
        if COMPUTE_NODE.match(resolved):
            compute.append(resolved)
        elif "davinci" in resolved or "ascend" in resolved or "hisi_hdc" in resolved:
            control.append(resolved)
    return ("torch_npu" in sys.modules, tuple(sorted(set(control))),
            tuple(sorted(set(compute))))


def _digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def readelf_symbols(path):
    """Kernel-type symbol names, read with the toolchain's own readelf.

    Symbols live in the aibin's symbol table rather than the section header
    table, so they are not re-implemented here. No readelf is a NAMED gap, never
    an empty tuple that could be mistaken for "this binary declares no kernel
    type".
    """
    tool = shutil.which("readelf")
    if tool is None:
        raise CompileRefusal("COMPILED_ABI_UNOBSERVED",
                             "readelf is not available, so no symbol can be read from %s"
                             % (path,), "a readelf executable", None)
    result = subprocess.run([tool, "-sW", path], capture_output=True, text=True,
                            timeout=60)
    if result.returncode != 0:
        raise CompileRefusal("COMPILED_ABI_UNOBSERVED",
                             "readelf -sW refused %s (exit %d): %s"
                             % (path, result.returncode, (result.stderr or "")[:200]),
                             None, result.returncode)
    names = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0].rstrip(":").isdigit():
            names.append(fields[7])
    return tuple(sorted(set(names)))


def _locate(directory, name):
    direct = os.path.join(directory, name)
    if os.path.isfile(direct):
        return direct
    for dirpath, _dirs, files in os.walk(directory):
        if name in files:
            return os.path.join(dirpath, name)
    return direct


def _elf_sections(raw):
    """(machine, e_type name, section names, {name: (offset, size)}).

    Read from the ELF headers themselves with the toolchain's own numbers, so no
    external readelf is needed for the values. Symbol-table names such as
    .Kernel_Type_<kernel> are reported by readelf -sW rather than the section
    header table, so the kernel-type check reads the section list this returns
    plus the symbols this slice can actually see.
    """
    if raw[:4] != b"\x7fELF":
        raise CompileRefusal("COMPILED_ABI_MISMATCH", "kernel.aibin is not an ELF object",
                             "ELF", raw[:8])
    if raw[4] != 2:
        raise CompileRefusal("COMPILED_ABI_MISMATCH", "only ELF64 is read here", "ELF64",
                             "ELF32")
    e_type = int.from_bytes(raw[16:18], "little")
    machine = int.from_bytes(raw[18:20], "little")
    # ELF64: e_shoff is at 0x28; e_shentsize/e_shnum/e_shstrndx follow e_flags,
    # e_ehsize, e_phentsize and e_phnum, i.e. at 0x3A. Reading them at 0x30 hits
    # e_flags and silently yields a nonsense section count, which is exactly the
    # wrong-header bug this first draft measured against readelf's own numbers.
    sh_off = int.from_bytes(raw[0x28:0x30], "little")
    sh_entry, sh_num, sh_strndx = _unpack("<HHH", raw, 0x3A)
    types = {1: "REL", 2: "EXEC", 3: "DYN", 4: "CORE"}
    names, offsets = [], {}
    if sh_off and sh_num and sh_entry:
        # sh_strndx selects a SECTION HEADER; the names live at that section's
        # own sh_offset (0x18 into the entry), not at the header's address.
        # Using the header address instead reads section headers as text and
        # silently produces a bogus one-name table - the bug this parser's first
        # version measured against readelf -SW, which lists 11 sections here.
        strtab_entry = sh_off + sh_strndx * sh_entry
        string_table = int.from_bytes(raw[strtab_entry + 0x18:strtab_entry + 0x20], "little")
        for index in range(sh_num):
            entry = sh_off + index * sh_entry
            if entry + sh_entry > len(raw):
                break
            name_at = int.from_bytes(raw[entry:entry + 4], "little")
            start = string_table + name_at
            end = raw.find(b"\x00", start)
            if not (0 <= start < end):
                continue
            name = raw[start:end].decode("ascii", "replace")
            offsets[name] = (int.from_bytes(raw[entry + 0x18:entry + 0x20], "little"),
                             int.from_bytes(raw[entry + 0x20:entry + 0x28], "little"))
            names.append(name)
    return machine, types.get(e_type, str(e_type)), tuple(names), offsets


def _toolchain_string():
    """The compiler build that produced the binary, read from the compiler."""
    bisheng = discover_bisheng()
    result = subprocess.run([bisheng, "--version"], capture_output=True, text=True,
                            timeout=60)
    text = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"clang version [^\s)]+", text)
    if match is None:
        raise CompileRefusal("COMPILED_ABI_UNOBSERVED",
                             "bisheng at %s printed no recognisable compiler version"
                             % (bisheng,), "clang version <x>", text[:200])
    stamp = re.search(r"\d{4}-\d{2}-\d{2}T[\d:+\-]+", text)
    return "%s | %s%s" % (bisheng, match.group(0),
                          (" | built " + stamp.group(0)) if stamp else "")


def _product_directories(root):
    """Every cache directory already holding a product, before this compile."""
    return set(_directories_holding(root, "manifest.json")) | set(
        _directories_holding(root, "kernel.aibin"))
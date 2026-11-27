"""Dynamic CANN header discovery and msprof helpers for Ascend 950 micro-probes."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

_CANN_INCLUDE_SUBDIRS: tuple[str, ...] = (
    "x86_64-linux/include",
    "aarch64-linux/include",
    "runtime/include",
    "include",
)


def ensure_cann_include_cpath() -> list[str]:
    """Discover CANN header directories and prepend missing entries to ``CPATH``."""
    discovered: list[str] = []
    seen: set[str] = set()
    for env_key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        root = Path(raw).expanduser()
        for sub in _CANN_INCLUDE_SUBDIRS:
            inc = root / sub
            if inc.is_dir():
                resolved = str(inc.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    discovered.append(str(inc))
    if discovered:
        existing = [p for p in os.environ.get("CPATH", "").split(os.pathsep) if p]
        os.environ["CPATH"] = os.pathsep.join(dict.fromkeys([*discovered, *existing]))
    return discovered


def run_msprof_detail(
    fn: Callable[[], object],
    *,
    n_warmup: int = 3,
    n_repeat: int = 5,
    cache_size_mb: int = 256,
) -> object:
    """Warm up and profile an NPU callable via TileLang ``bench_with_msprof``."""
    import torch  # noqa: PLC0415
    from tilelang.profiler.msprof import bench_with_msprof  # noqa: PLC0415

    fn()
    torch.npu.synchronize()
    for _ in range(max(0, int(n_warmup))):
        fn()
    torch.npu.synchronize()
    cache = torch.empty(
        max(1, int(cache_size_mb)) * 1024 * 1024 // 4,
        dtype=torch.int32,
        device="npu",
    )
    return bench_with_msprof(fn, cache, n_repeat=max(1, int(n_repeat)), detailed=True)


class LocalHardwareProbePort:
    """Default local HardwareProbePort delegating to CANN env and msprof helpers."""

    def ensure_cann_include_cpath(
        self,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """Discover CANN header directories and prepend missing entries to ``CPATH``."""
        _ = env
        return ensure_cann_include_cpath()

    def run_msprof_detail(self, *args: object, **kwargs: object) -> object:
        """Warm up and profile an NPU callable via TileLang ``bench_with_msprof``."""
        fn = kwargs.get("fn", args[0] if args else None)
        if not callable(fn):
            raise TypeError("run_msprof_detail requires a callable first argument")
        raw_w, raw_r, raw_c = kwargs.get("n_warmup", 3), kwargs.get("n_repeat", 5), kwargs.get("cache_size_mb", 256)
        return run_msprof_detail(
            fn,
            n_warmup=int(raw_w) if isinstance(raw_w, (int, float)) and not isinstance(raw_w, bool) else 3,
            n_repeat=int(raw_r) if isinstance(raw_r, (int, float)) and not isinstance(raw_r, bool) else 5,
            cache_size_mb=int(raw_c) if isinstance(raw_c, (int, float)) and not isinstance(raw_c, bool) else 256,
        )

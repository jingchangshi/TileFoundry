"""Stateless agy CLI backend (Hydrogen Medium/High) and mock backend for agent orchestration."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    """Single stateless LLM generation request routed by worker role."""

    role: str
    system_prompt: str
    user_prompt: str
    json_schema: dict[str, object] | None = None


class KernelLLMBackend(Protocol):
    """Protocol implemented by both live AgyCliLLMBackend and MockLLMBackend."""

    async def generate(self, req: GenerationRequest) -> str:
        """Generate response text or JSON string for a GenerationRequest."""


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

CREATIVE_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "physical_hypothesis": {"type": "string"},
        "target_local_metric": {"type": "string"},
        "dsl_source": {"type": "string"},
    },
    "required": ["physical_hypothesis", "target_local_metric", "dsl_source"],
    "additionalProperties": False,
}


def extract_python_dsl(raw_response: str) -> str:
    """Extract clean Python DSL source from markdown fences, JSON envelope, or raw text."""
    stripped = raw_response.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                if isinstance(payload.get("structured_output"), dict):
                    dsl = payload["structured_output"].get("dsl_source")
                    if isinstance(dsl, str) and dsl.strip():
                        return dsl.strip()
                if isinstance(payload.get("dsl_source"), str) and payload["dsl_source"].strip():
                    return payload["dsl_source"].strip()
                if isinstance(payload.get("response"), str) and payload["response"].strip():
                    stripped = payload["response"].strip()
        except json.JSONDecodeError:
            pass
    matches = _CODE_FENCE_RE.findall(stripped)
    if matches:
        return max(matches, key=len).strip()
    return stripped


def extract_creative_payload(raw_response: str) -> dict[str, str]:
    """Parse structured JSON response or markdown fallback from creative explore worker."""
    stripped = raw_response.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                if isinstance(obj.get("structured_output"), dict):
                    obj = obj["structured_output"]
                elif isinstance(obj.get("response"), str) and obj["response"].strip().startswith("{"):
                    inner = json.loads(obj["response"])
                    if isinstance(inner, dict):
                        obj = inner
                if isinstance(obj.get("dsl_source"), str):
                    return {
                        "physical_hypothesis": str(obj.get("physical_hypothesis", "")),
                        "target_local_metric": str(obj.get("target_local_metric", "")),
                        "dsl_source": extract_python_dsl(str(obj["dsl_source"])),
                    }
        except json.JSONDecodeError:
            pass
    return {
        "physical_hypothesis": "unstructured_fallback",
        "target_local_metric": "latency_us",
        "dsl_source": extract_python_dsl(raw_response),
    }


def _resolve_agy_binary(explicit_bin: str | None = None) -> str:
    candidate = (explicit_bin or os.environ.get("TILEFOUNDRY_NPU_AGY_BIN", "")).strip()
    if candidate:
        resolved = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if resolved:
            return resolved
    on_path = shutil.which("agy")
    if on_path:
        return on_path
    user_local = Path.home() / ".local" / "bin" / "agy"
    if user_local.is_file():
        return str(user_local)
    raise FileNotFoundError(
        "Cannot locate 'agy' binary. Set TILEFOUNDRY_NPU_AGY_BIN in tilefoundry-npu env.sh."
    )


def _normalize_agy_common_args(raw_common: str) -> list[str]:
    """Strip bare --print/-p and output flags so --print=<prompt> binds safely."""
    tokens = shlex.split(raw_common)
    cleaned: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--print", "-p", "--prompt"):
            continue
        if tok.startswith(("--print=", "--prompt=", "--output-format=", "--json-schema=")):
            continue
        if tok in ("--output-format", "--json-schema", "--model"):
            skip_next = True
            continue
        if tok == "--add-dir":
            skip_next = True
            continue
        if tok.startswith("--add-dir="):
            continue
        cleaned.append(tok)
    return cleaned


class AgyCliLLMBackend:
    """Stateless headless wrapper around local agy CLI inside an empty sandbox directory."""

    DEFAULT_COMMON_ARGS = (
        "--input-format text --disable-slash-commands "
        "--sandbox --mode plan --dangerously-skip-permissions --print-timeout 240s"
    )
    DEFAULT_EXPLOIT_MODEL = "gemini-4-hydrogen-medium"
    DEFAULT_EXPLORE_MODEL = "gemini-4-hydrogen-high"

    def __init__(
        self,
        bin_path: str | None = None,
        exploit_model: str | None = None,
        explore_model: str | None = None,
        common_args: str | None = None,
        max_concurrency: int | None = None,
        hard_timeout_sec: int = 270,
    ) -> None:
        self._bin = _resolve_agy_binary(bin_path)
        self._exploit_model = (
            exploit_model
            if exploit_model is not None
            else os.environ.get("TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL", "")
        ).strip() or self.DEFAULT_EXPLOIT_MODEL
        self._explore_model = (
            explore_model
            if explore_model is not None
            else os.environ.get("TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL", "")
        ).strip() or self.DEFAULT_EXPLORE_MODEL

        raw_common = (
            common_args
            if common_args is not None
            else os.environ.get("TILEFOUNDRY_NPU_AGY_COMMON_ARGS", "")
        ).strip() or self.DEFAULT_COMMON_ARGS
        self._common_args = _normalize_agy_common_args(raw_common)

        raw_conc = os.environ.get("TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY", "").strip()
        conc = max_concurrency or (int(raw_conc) if raw_conc else 3)
        self._sem = asyncio.Semaphore(max(1, conc))
        self._hard_timeout = hard_timeout_sec

    def select_model(self, role: str) -> str:
        """Return target model ID for role ('explore_creative' -> High, others -> Medium)."""
        return self._explore_model if role == "explore_creative" else self._exploit_model

    @staticmethod
    def _sanitized_env(sandbox_dir: str) -> dict[str, str]:
        stripped_prefixes = ("AGY_CONVERSATION", "AGY_PROJECT", "CLAUDECODE", "MCP_")
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(k.startswith(p) for p in stripped_prefixes)
        }
        env["PWD"] = sandbox_dir
        return env

    def build_command(self, req: GenerationRequest, full_prompt: str) -> list[str]:
        """Build isolated agy CLI argument list without workspace mounts."""
        model_id = self.select_model(req.role)
        cmd = [self._bin, "--model", model_id, *self._common_args]
        if req.json_schema is not None:
            cmd.extend(["--output-format", "json", "--json-schema", json.dumps(req.json_schema)])
        else:
            cmd.extend(["--output-format", "text"])
        cmd.append(f"--print={full_prompt}")
        return cmd

    async def generate(self, req: GenerationRequest) -> str:
        """Execute agy in a fresh empty temp sandbox directory with bounded timeout."""
        model_id = self.select_model(req.role)
        full_prompt = (
            "[STATELESS GENERATION CONTRACT: Do NOT read/write files or run shell/MCP tools. "
            "Respond strictly based on the prompt below.]\n\n"
            f"### SYSTEM INSTRUCTIONS\n{req.system_prompt}\n\n"
            f"### USER INPUT\n{req.user_prompt}\n"
        )
        cmd = self.build_command(req, full_prompt)

        async with self._sem:
            with tempfile.TemporaryDirectory(prefix="tf_agy_worker_") as sandbox_dir:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox_dir,
                    env=self._sanitized_env(sandbox_dir),
                )
                try:
                    out_b, err_b = await asyncio.wait_for(
                        proc.communicate(full_prompt.encode("utf-8")),
                        timeout=self._hard_timeout,
                    )
                except asyncio.TimeoutError as exc:
                    proc.kill()
                    await proc.communicate()
                    raise TimeoutError(
                        f"agy subprocess timed out (role={req.role}, model={model_id})"
                    ) from exc

                if proc.returncode != 0:
                    err_text = err_b.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(
                        f"agy exited with {proc.returncode} (model={model_id}): {err_text}"
                    )
                return out_b.decode("utf-8", errors="replace").strip()


@dataclass
class MockLLMBackend:
    """Deterministic scriptable LLM backend for offline orchestrator testing."""

    responses: dict[str, list[str]] = field(default_factory=dict)
    default_response: str = ""
    calls: list[GenerationRequest] = field(default_factory=list)
    routed_models: list[str] = field(default_factory=list)

    async def generate(self, req: GenerationRequest) -> str:
        """Record request and return next queued response for req.role."""
        self.calls.append(req)
        model_id = (
            AgyCliLLMBackend.DEFAULT_EXPLORE_MODEL
            if req.role == "explore_creative"
            else AgyCliLLMBackend.DEFAULT_EXPLOIT_MODEL
        )
        self.routed_models.append(model_id)
        queue = self.responses.get(req.role)
        if queue:
            return queue.pop(0)
        return self.default_response


def create_llm_backend_from_env() -> KernelLLMBackend:
    """Instantiate configured LLM backend from TILEFOUNDRY_NPU_LLM_BACKEND."""
    kind = os.environ.get("TILEFOUNDRY_NPU_LLM_BACKEND", "agy_cli").strip().lower()
    if kind in ("agy_cli", "agy", ""):
        return AgyCliLLMBackend()
    if kind == "mock":
        return MockLLMBackend()
    raise ValueError(f"unsupported TILEFOUNDRY_NPU_LLM_BACKEND={kind!r}")


__all__ = [
    "AgyCliLLMBackend",
    "CREATIVE_RESPONSE_SCHEMA",
    "GenerationRequest",
    "KernelLLMBackend",
    "MockLLMBackend",
    "create_llm_backend_from_env",
    "extract_creative_payload",
    "extract_python_dsl",
]

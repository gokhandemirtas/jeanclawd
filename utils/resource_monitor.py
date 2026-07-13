"""Background thread that logs VRAM and system memory usage every N seconds."""

import os
import sys
import threading
from functools import lru_cache

import psutil

from utils.log import logger


def detect_memory_gb() -> float:
  """Return the most relevant memory pool in GB for this backend.

  Priority: NVIDIA VRAM → Apple unified RAM → system RAM.
  Falls back to 8.0 GB on any detection failure.
  """
  # NVIDIA VRAM (Linux / llamacpp + CUDA)
  try:
    from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
    nvmlInit()
    info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
    return info.total / 1024 ** 3
  except Exception:
    pass

  # Apple Silicon unified memory (MLX) or CPU-only — total RAM is the pool
  try:
    return psutil.virtual_memory().total / 1024 ** 3
  except Exception:
    return 8.0


# MIN and MAX are exposed so callers can show them in debug output.
CHAT_HISTORY_MIN = 4
CHAT_HISTORY_MAX = 20


@lru_cache(maxsize=1)
def _get_encoder():
  """tiktoken encoder. Cached at module level — first call pays the ~50ms load.

  cl100k_base is the same encoding the /ctx command uses (commands.py:167).
  It's an approximation for non-OpenAI models (Qwen3 has its own BPE) but is
  good enough for a real-time meter and stays consistent with /ctx.
  """
  import tiktoken
  return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
  """Approximate token count of a string using cl100k_base."""
  if not text:
    return 0
  return len(_get_encoder().encode(text))


# Updated by LLM-consuming code (e.g. skills/conversation/skill.py) just before
# issuing a request. The background monitor thread reads this each tick and
# appends a used/available meter to its status line. Plain int — single GIL
# write is atomic, no lock needed.
_context_used_tokens: int = 0
_context_max_tokens: int | None = None


def set_context_used(used: int) -> None:
  """Push the current context-window token count for display in the monitor line."""
  global _context_used_tokens
  _context_used_tokens = max(0, used)


def get_context_max() -> int | None:
  """Best-effort lookup of the model's max context length (cached after first call)."""
  global _context_max_tokens
  if _context_max_tokens is not None:
    return _context_max_tokens
  try:
    from providers import get_provider
    _context_max_tokens = get_provider().context_length()
  except Exception:
    return None
  return _context_max_tokens


CONTEXT_FILE = "logs/context.json"


def push_context_to_file() -> None:
  """Write the current context state to logs/context.json for monitor.py to read.

  Uses a temp file + os.replace for atomicity so the Textual monitor (which
  polls at 1Hz) never sees a half-written file. Failures are swallowed — the
  meter is observability, not correctness.
  """
  import json
  import tempfile

  try:
    payload = {
      "used": _context_used_tokens,
      "max": _context_max_tokens,
      "ts": __import__("time").time(),
    }
    dir_name = os.path.dirname(CONTEXT_FILE) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, prefix=".context.", suffix=".tmp")
    try:
      with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
      os.replace(tmp, CONTEXT_FILE)
    except Exception:
      try:
        os.unlink(tmp)
      except OSError:
        pass
      raise
  except Exception as e:
    logger.debug(f"[resource_monitor] context push failed: {e}")


@lru_cache(maxsize=1)
def compute_chat_history_limit() -> int:
  """Return how many history messages to pass to the chat model.

  Scales between CHAT_HISTORY_MIN and CHAT_HISTORY_MAX using two signals:
  - Available memory: more memory → larger model → better long-range coherence
  - Actual model context length queried from the provider at startup

  Computed once per process (lru_cache) since hardware doesn't change at runtime.
  """
  from providers import get_provider

  memory_gb = detect_memory_gb()

  try:
    ctx = get_provider().context_length()
    logger.debug(f"[resource_monitor] Model context length: {ctx} tokens")
  except Exception:
    ctx = int(os.environ.get("MODEL_CTX", 8192))
    logger.warning(f"[resource_monitor] Could not query model context length, falling back to MODEL_CTX={ctx}")

  # Context cap: reserve ~2000 tokens for system prompt + user query + response headroom.
  # Rough estimate: 150 tokens per history message.
  ctx_cap = max(CHAT_HISTORY_MIN, (ctx - 2000) // 150)

  # Memory cap: 0.7 msgs/GB is a rough heuristic calibrated so that:
  #   8 GB  →  6 msgs  (3B model on M1 base / RTX 3060)
  #   16 GB → 11 msgs  (7B model on M2 Pro / RTX 4080)
  #   24 GB → 17 msgs  (13B model on M2 Max / RTX 3090)
  mem_cap = round(memory_gb * 0.7)

  limit = max(CHAT_HISTORY_MIN, min(CHAT_HISTORY_MAX, ctx_cap, mem_cap))
  logger.debug(f"[resource_monitor] Chat history limit: {limit} (ctx_cap={ctx_cap}, mem_cap={mem_cap})")
  return limit


def _monitor_loop(interval: float, stop_event: threading.Event):
  """Periodically log GPU VRAM and system RAM usage."""
  from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
  nvmlInit()
  gpu_handle = nvmlDeviceGetHandleByIndex(0)
  process = psutil.Process(os.getpid())

  while not stop_event.wait(interval):
    # GPU VRAM
    gpu_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_used = gpu_info.used / 1024**2
    gpu_total = gpu_info.total / 1024**2

    # System RAM (process-level)
    mem_info = process.memory_info()
    rss = mem_info.rss / 1024**2

    # System RAM (overall)
    sys_mem = psutil.virtual_memory()
    sys_used = sys_mem.used / 1024**2
    sys_total = sys_mem.total / 1024**2

    line = (
      f"[monitor] VRAM: {gpu_used:.0f}/{gpu_total:.0f}MB | "
      f"RAM (process): {rss:.0f}MB | "
      f"RAM (system): {sys_used:.0f}/{sys_total:.0f}MB"
    )
    print(f"\r{line:<80}", end="", flush=True, file=sys.stderr)


_stop_event: threading.Event | None = None
_thread: threading.Thread | None = None


def start_monitor(interval: float = 1.0):
  """Start the background resource monitor.

  Args:
      interval: Seconds between reports (default: 1.0)
  """
  global _stop_event, _thread
  if _thread is not None and _thread.is_alive():
    logger.warning("Resource monitor already running")
    return

  _stop_event = threading.Event()
  _thread = threading.Thread(
    target=_monitor_loop,
    args=(interval, _stop_event),
    daemon=True,
    name="resource-monitor",
  )
  _thread.start()
  from utils.log import logger
  logger.info(f"Resource monitor started (interval={interval}s)")


def stop_monitor():
  """Stop the background resource monitor."""
  global _stop_event, _thread
  if _stop_event is not None:
    _stop_event.set()
  if _thread is not None:
    _thread.join(timeout=5)
    _thread = None
  _stop_event = None
  from utils.log import logger
  logger.info("Resource monitor stopped")


if __name__ == "__main__":
  import subprocess
  subprocess.run([sys.executable, str(Path(__file__).parent.parent / "monitor.py")] + sys.argv[1:])

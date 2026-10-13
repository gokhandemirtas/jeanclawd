"""Shared display helpers for monitor.py and tui.py.

Any symbol used by both the standalone MonitorApp and the TUI dashboard
lives here so neither file duplicates the other.
"""

from __future__ import annotations

import json

import psutil
from rich.markup import escape

from utils.resource_monitor import CONTEXT_FILE

# ── log level styles ──────────────────────────────────────────────────────────

LEVEL_STYLE: dict[str, str] = {
    "DEBUG": "dim white",
    "INFO":  "bold cyan",
    "WARN":  "bold yellow",
    "ERROR": "bold red",
}
ALL_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARN", "ERROR")

# ── bar / formatting helpers ──────────────────────────────────────────────────


def rbar(used: float, total: float, width: int = 16) -> str:
    pct = used / total if total else 0
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)


def fmt_mb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def fmt_metric(value, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


# ── LLM performance entry ─────────────────────────────────────────────────────


def performance_text(entry: dict) -> str:
    model      = escape(str(entry.get("model", "unknown")))
    prompt     = fmt_metric(entry.get("prompt_tokens"))
    completion = fmt_metric(entry.get("completion_tokens"))
    server_tps = fmt_metric(entry.get("server_tokens_per_second"), " tok/s")
    meas_tps   = fmt_metric(entry.get("measured_tokens_per_second"), " tok/s")
    srv_ttft   = fmt_metric(entry.get("server_ttft_ms"), " ms")
    cli_ttft   = fmt_metric(entry.get("client_ttft_ms"), " ms")
    return (
        f"[dim]{entry.get('ts', '')}[/dim] [bold yellow]LLM PERF[/bold yellow] "
        f"[bold]{model}[/bold]\n"
        f"  tokens: prompt={prompt}, completion={completion}  "
        f"throughput: server={server_tps}, measured={meas_tps}  "
        f"TTFT: server={srv_ttft}, client={cli_ttft}"
    )


# ── system resource markup ────────────────────────────────────────────────────


def _meter(label: str, value_str: str, bar_filled: int, bar_total: int, bar_color: str) -> list[str]:
    """Render a two-line resource meter with a consistent value and bar."""
    filled = min(bar_filled, bar_total)
    empty = bar_total - filled
    bar_line = f"[{bar_color}]{'█' * filled}[/{bar_color}][dim]{'░' * empty}[/dim]"
    lines = [
        f"[bold {bar_color}]{label}[/bold {bar_color}]  [bold]{value_str}[/bold]",
        bar_line,
    ]
    lines.append("")
    return lines


def resource_markup(*, compact: bool = False, bar_width: int = 16) -> str:
    """Render resource meters, optionally sized to a dashboard side rail."""
    bar_width = max(8, bar_width)
    lines: list[str] = [
        "[bold #22d3ee]RESOURCES[/bold #22d3ee]  [dim]POLL: 1s[/dim]",
        f"[dim]{'─' * bar_width}[/dim]",
        "",
    ]

    def usage(used: float, total: float, percent: float) -> str:
        if compact:
            return f"{used / 1024:.1f}/{total / 1024:.1f} GB {percent:.0f}%"
        return f"{fmt_mb(used)} / {fmt_mb(total)} ({percent:.0f}%)"

    # CPU
    pct = psutil.cpu_percent(interval=None)
    filled = int(bar_width * pct / 100)
    lines += _meter("CPU", f"{pct:.1f}%", filled, bar_width, "#22d3ee")

    # RAM: derive the used value from `available` so it agrees with psutil.percent.
    vm = psutil.virtual_memory()
    used_mb  = (vm.total - vm.available) / 1024 ** 2
    total_mb = vm.total / 1024 ** 2
    filled = int(bar_width * vm.percent / 100)
    lines += _meter("RAM", usage(used_mb, total_mb, vm.percent), filled, bar_width, "#7c3aed")

    # Swap
    try:
        sw = psutil.swap_memory()
        if sw.total > 0:
            sw_used  = sw.used  / 1024 ** 2
            sw_total = sw.total / 1024 ** 2
            sw_pct   = sw.percent
            filled   = int(bar_width * sw_pct / 100)
            lines += _meter("Swap", usage(sw_used, sw_total, sw_pct), filled, bar_width, "#22d3ee")
    except OSError:
        pass

    # Disk
    try:
        du = psutil.disk_usage("/")
        free_gb  = du.free  / 1024 ** 3
        total_gb = du.total / 1024 ** 3
        # On APFS, `used` can be small while free space reflects the shared volume.
        # Show available capacity and use the matching available-space percentage.
        free_pct = du.free / du.total * 100 if du.total else 0
        filled   = int(bar_width * free_pct / 100)
        disk_value = (
            f"{free_gb:.0f}/{total_gb:.0f} GB {free_pct:.0f}%"
            if compact
            else f"{free_gb:.0f} GB free / {total_gb:.0f} GB ({free_pct:.0f}%)"
        )
        lines += _meter("Disk", disk_value, filled, bar_width, "#22d3ee")
    except Exception:
        pass

    # NVIDIA VRAM
    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
        )
        nvmlInit()
        h    = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        vram_used  = info.used  / 1024 ** 2
        vram_total = info.total / 1024 ** 2
        vram_pct   = vram_used / vram_total * 100 if vram_total else 0
        filled     = int(bar_width * vram_pct / 100)
        lines += _meter("VRAM", usage(vram_used, vram_total, vram_pct), filled, bar_width, "#facc15")
    except Exception:
        pass

    # LLM context window
    used_ctx = max_ctx = 0
    has_ctx = False
    try:
        if CONTEXT_FILE.exists():
            ctx = json.loads(CONTEXT_FILE.read_text())
            used_ctx = int(ctx.get("used", 0))
            max_ctx  = int(ctx.get("max",  0)) or 0
            has_ctx  = True
    except Exception:
        pass

    if not has_ctx or max_ctx <= 0:
        lines += _meter("CONTEXT", "— waiting", 0, bar_width, "#4ade80")
    else:
        pct_ctx = used_ctx / max_ctx * 100
        display_pct = max(1, round(pct_ctx)) if used_ctx else 0
        filled = max(1, int(bar_width * pct_ctx / 100)) if display_pct else 0
        ctx_value = (
            f"{used_ctx / 1000:.0f}k/{max_ctx / 1000:.0f}k {display_pct}%"
            if compact
            else f"{used_ctx:,} / {max_ctx:,} tokens ({display_pct}%)"
        )
        lines += _meter("CONTEXT", ctx_value, filled, bar_width, "#4ade80")

    return "\n".join(lines)

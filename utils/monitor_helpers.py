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


def resource_markup() -> str:
    lines: list[str] = []

    # CPU
    pct = psutil.cpu_percent(interval=None)
    bar = rbar(pct, 100)
    lines.append("[bold cyan]CPU[/bold cyan]")
    lines.append(f"[dim]{bar}[/dim] [bold]{pct:.1f}%[/bold]")
    lines.append("")

    # RAM
    vm = psutil.virtual_memory()
    used_mb  = vm.used  / 1024 ** 2
    total_mb = vm.total / 1024 ** 2
    bar = rbar(used_mb, total_mb)
    lines.append("[bold cyan]RAM[/bold cyan]")
    lines.append(f"[dim]{bar}[/dim]")
    lines.append(f"[bold]{fmt_mb(used_mb)}[/bold] / {fmt_mb(total_mb)}")
    lines.append("")

    # Swap
    try:
        sw = psutil.swap_memory()
        if sw.total > 0:
            sw_used  = sw.used  / 1024 ** 2
            sw_total = sw.total / 1024 ** 2
            bar = rbar(sw_used, sw_total)
            lines.append("[bold cyan]Swap[/bold cyan]")
            lines.append(f"[dim]{bar}[/dim]")
            lines.append(f"[bold]{fmt_mb(sw_used)}[/bold] / {fmt_mb(sw_total)}")
            lines.append("")
    except OSError:
        pass

    # Disk
    try:
        du = psutil.disk_usage("/")
        bar = rbar(du.used, du.total)
        free_gb  = du.free  / 1024 ** 3
        total_gb = du.total / 1024 ** 3
        lines.append("[bold cyan]Disk[/bold cyan]")
        lines.append(f"[dim]{bar}[/dim]")
        lines.append(f"[bold]{free_gb:.1f} GB[/bold] free / {total_gb:.0f} GB")
        lines.append("")
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
        name = nvmlDeviceGetName(h)
        vram_used  = info.used  / 1024 ** 2
        vram_total = info.total / 1024 ** 2
        bar = rbar(vram_used, vram_total)
        lines.append(f"[bold cyan]VRAM[/bold cyan] [dim]{name}[/dim]")
        lines.append(f"[dim]{bar}[/dim]")
        lines.append(f"[bold]{fmt_mb(vram_used)}[/bold] / {fmt_mb(vram_total)}")
        lines.append("")
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

    lines.append("[bold cyan]Context[/bold cyan] [dim]window[/dim]")
    if not has_ctx or max_ctx <= 0:
        lines.append(f"[dim]{rbar(0, 1)}[/dim]")
        lines.append("[dim]— waiting for first request[/dim]")
    else:
        bar     = rbar(used_ctx, max_ctx)
        pct_ctx = used_ctx / max_ctx * 100
        lines.append(f"[dim]{bar}[/dim]")
        lines.append(f"[bold]{used_ctx:,}[/bold] / {max_ctx:,} ({pct_ctx:.0f}%)")

    return "\n".join(lines)

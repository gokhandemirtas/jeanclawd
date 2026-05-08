import json
import os
import platform
import shutil
import subprocess

import psutil
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

_console = Console()


def _fmt_gb(b: int) -> str:
	return f"{b / 1024**3:.1f} GB"


def _models() -> list[tuple[str, str]]:
	keys = [
		("Generalist", "MODEL_GENERALIST"),
		("Embedding", "MODEL_EMBEDDING"),
		("Host", "MODEL_HOST"),
	]
	return [(label, os.getenv(env, "—")) for label, env in keys]


def _backend() -> list[tuple[str, str]]:
	system = platform.system()
	rows: list[tuple[str, str]] = [
		("OS", f"{system} {platform.release()} ({platform.machine()})"),
	]

	if system == "Darwin":
		try:
			raw = subprocess.run(
				["system_profiler", "SPHardwareDataType", "-json"],
				capture_output=True,
				text=True,
				timeout=5,
			)
			hw = json.loads(raw.stdout)["SPHardwareDataType"][0]
			chip = hw.get("chip_type") or hw.get("cpu_type", "")
			if chip:
				rows.append(("Chip", chip))
		except Exception:
			pass

		try:
			raw = subprocess.run(
				["system_profiler", "SPDisplaysDataType", "-json"],
				capture_output=True,
				text=True,
				timeout=5,
			)
			for entry in json.loads(raw.stdout)["SPDisplaysDataType"]:
				cores = entry.get("sppci_cores")
				if cores:
					rows.append(("GPU Cores", cores))
					break
		except Exception:
			pass

	else:
		try:
			out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
			for line in out.stdout.splitlines():
				if line.startswith("Model name"):
					rows.append(("CPU", line.split(":", 1)[1].strip()))
					break
		except Exception:
			pass

	return rows


def _resources() -> list[tuple[str, str, str]]:
	system = platform.system()
	vm = psutil.virtual_memory()
	rows: list[tuple[str, str, str]] = []

	if system == "Darwin":
		rows.append(("Unified Memory", _fmt_gb(vm.total), _fmt_gb(vm.available)))
	else:
		rows.append(("RAM", _fmt_gb(vm.total), _fmt_gb(vm.available)))

		if shutil.which("nvidia-smi"):
			try:
				out = subprocess.run(
					[
						"nvidia-smi",
						"--query-gpu=name,memory.total,memory.free",
						"--format=csv,noheader,nounits",
					],
					capture_output=True,
					text=True,
					timeout=5,
				)
				for line in out.stdout.strip().splitlines():
					parts = [p.strip() for p in line.split(",")]
					if len(parts) == 3:
						name, total_mib, free_mib = parts
						rows.append(
							(
								f"VRAM ({name})",
								f"{int(total_mib) / 1024:.1f} GB",
								f"{int(free_mib) / 1024:.1f} GB",
							)
						)
			except Exception:
				pass

		elif shutil.which("rocm-smi"):
			try:
				out = subprocess.run(
					["rocm-smi", "--showmeminfo", "vram", "--json"],
					capture_output=True,
					text=True,
					timeout=5,
				)
				data = json.loads(out.stdout)
				for gpu_id, gpu_data in data.items():
					total = int(gpu_data.get("VRAM Total Memory (B)", 0))
					used = int(gpu_data.get("VRAM Total Used Memory (B)", 0))
					if total:
						rows.append((f"VRAM (AMD {gpu_id})", _fmt_gb(total), _fmt_gb(total - used)))
			except Exception:
				pass

	return rows


def boot() -> None:
	models_table = Table(
		box=None, show_header=True, header_style="bold", padding=(0, 2, 0, 0)
	)
	models_table.add_column("Role", style="dim", min_width=20)
	models_table.add_column("Model")
	for role, model in _models():
		models_table.add_row(role, f"[cyan]{model}[/cyan]")

	backend_table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
	backend_table.add_column("Key", style="dim", min_width=12)
	backend_table.add_column("Value")
	for key, val in _backend():
		backend_table.add_row(key, val)

	res_table = Table(
		box=None, show_header=True, header_style="bold", padding=(0, 2, 0, 0)
	)
	res_table.add_column("Resource", style="dim", min_width=18)
	res_table.add_column("Total", justify="right")
	res_table.add_column("Free", justify="right", style="green")
	for name, total, free in _resources():
		res_table.add_row(name, total, free)

	_console.print()
	_console.print(Rule("[bold magenta]AIDA[/bold magenta]"))
	_console.print(Panel(models_table, title="Models", border_style="dim"))
	_console.print(
		Columns(
			[
				Panel(backend_table, title="Backend", border_style="dim"),
				Panel(res_table, title="Resources", border_style="dim"),
			]
		)
	)
	_console.print()

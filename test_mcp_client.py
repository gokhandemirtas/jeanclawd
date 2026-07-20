"""MCP client smoke test: exercise multiple tools to prove the connection."""
import asyncio
import json
import logging
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
	server = StdioServerParameters(
		command="uv",
		args=["--directory", str(Path(__file__).parent.resolve()), "run", "mcp_server.py"],
	)
	async with stdio_client(server) as (read, write):
		async with ClientSession(read, write) as session:
			await session.initialize()
			print("[client] connected to jeanclawd MCP server\n", file=sys.stderr)

			tests = [
				("list_personas", {}, "Meta tool, no DB needed"),
				("show_context", {}, "Meta tool, no DB needed"),
				("list_memories", {"limit": 5}, "Memory tool, needs DB"),
				("list_notes", {}, "Notes tool, needs DB"),
				("weather", {"query": "weather in London"}, "Weather skill (has known intent-classifier bug)"),
			]
			for name, args, note in tests:
				print(f"[client] -> {name}({args})  // {note}", file=sys.stderr)
				try:
					result = await session.call_tool(name, args)
					for block in result.content:
						if hasattr(block, "text"):
							preview = block.text[:300]
							print(f"  [OK]  {preview}", file=sys.stderr)
				except Exception as e:
					print(f"  [ERR] {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)


if __name__ == "__main__":
	logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s", stream=sys.stderr)
	asyncio.run(main())

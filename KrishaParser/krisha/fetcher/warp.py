"""Opt-in reconnect of the installed WARP local proxy, never a system VPN mode."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


def enabled(proxy: dict[str, str]) -> bool:
    return (os.getenv("KRISHA_WARP_RECONNECT") == "1"
            and proxy.get("server") == "http://127.0.0.1:40000")


def reconnect() -> None:
    executable = Path(os.getenv("ProgramFiles", "C:/Program Files")) / "Cloudflare/Cloudflare WARP/warp-cli.exe"

    def call(*args: str) -> str:
        try:
            result = subprocess.run([str(executable), "--accept-tos", *args],
                                    capture_output=True, text=True, timeout=15, check=True,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return result.stdout
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError("WARP command failed; check the local proxy service") from None

    if "WarpProxy on port 40000" not in call("settings"):
        raise RuntimeError("WARP reconnect requires local proxy mode on port 40000")
    call("disconnect")
    call("connect")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if json.loads(call("--json", "status")).get("status") == "Connected":
            return
        time.sleep(1)
    raise RuntimeError("WARP did not reconnect within 30 seconds")

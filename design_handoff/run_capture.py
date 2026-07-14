"""Custom capture runner that points Playwright at an HTTP URL rather than a file://
path. Mirrors capture.py logic verbatim; only the navigation target changes because
Babel-standalone's XHR for the .jsx src= files is blocked on file:// by Chromium CORS.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

URL = "http://localhost:8765/record.html"
OUT = Path(__file__).parent / "AllowanceMap_Animation.mp4"
FPS = 30
WIDTH = 1920
HEIGHT = 1080
SELECTOR = "#root"
CRF = 18
PRESET = "medium"
READY_TIMEOUT_MS = 30000
SETTLE_MS = 800

tempdir_ctx = tempfile.TemporaryDirectory(prefix="allowancemap_frames_")
frames_dir = Path(tempdir_ctx.name)
print(f"[capture] loading {URL}")

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--hide-scrollbars", "--force-device-scale-factor=1"])
    ctx = browser.new_context(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
    page = ctx.new_page()
    page.on("pageerror", lambda e: print(f"[pageerror] {e}"))
    page.goto(URL, wait_until="networkidle")
    page.wait_for_function("() => window.__ready === true", timeout=READY_TIMEOUT_MS)
    try:
        page.evaluate("() => document.fonts && document.fonts.ready")
    except Exception:
        pass
    page.wait_for_timeout(SETTLE_MS)
    try:
        page.evaluate("() => { try { localStorage.clear(); sessionStorage.clear(); } catch(e){} }")
    except Exception:
        pass

    duration = float(page.evaluate("() => window.__duration"))
    total_frames = int(round(FPS * duration))
    print(f"[capture] {total_frames} frames ({duration:.2f}s @ {FPS}fps) -> {frames_dir}")

    target = page.locator(SELECTOR)
    if target.count() == 0:
        raise SystemExit(f"Selector not found on page: {SELECTOR}")

    for i in range(total_frames + 1):
        t = i / FPS
        page.evaluate(f"window.__setTime({t})")
        page.evaluate(
            "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
        )
        out_png = frames_dir / f"frame_{i:05d}.png"
        target.screenshot(path=str(out_png))
        if i % FPS == 0:
            print(f"  frame {i}/{total_frames} (t={t:.2f}s)")

    browser.close()

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
print(f"[encode] {ffmpeg}")
OUT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([
    ffmpeg, "-y",
    "-framerate", str(FPS),
    "-i", str(frames_dir / "frame_%05d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", str(CRF),
    "-preset", PRESET,
    "-movflags", "+faststart",
    str(OUT),
], check=True)
print(f"[done] wrote {OUT} ({OUT.stat().st_size/1024:.1f} KiB)")
tempdir_ctx.cleanup()

"""Render the commercial-style replay video: python -m src.simulation.make_video [--frames 90,270] [--method residual_rl]

Serves outputs/sumo_viewer/ locally, opens it in headless Edge (or Chromium) via Playwright in director mode,
renders every frame deterministically (director.js), waits for the satellite tiles, screenshots, then encodes
an H.264 MP4 with the ffmpeg binary that ships with imageio-ffmpeg. Output: outputs/sumo_viewer/dubai_ai_taxi.mp4
"""
from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "outputs" / "sumo_viewer"


def serve(port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(VIEWER))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def render(method: str, frames: list[int] | None, out_dir: Path, width=1920, height=1080, port=8767) -> int:
    from playwright.sync_api import sync_playwright
    for name in ("viewer.html", "director.js"):                      # always render the current viewer code
        shutil.copy(ROOT / "src" / "simulation" / name, VIEWER / ("index.html" if name == "viewer.html" else name))
    out_dir.mkdir(parents=True, exist_ok=True)
    srv = serve(port)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge", headless=True)
        except Exception:
            browser = p.chromium.launch(headless=True)
        page = None

        def open_page():
            pg = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            pg.goto(f"http://127.0.0.1:{port}/#director=1&m={method}", wait_until="networkidle")
            pg.wait_for_function("window.directorInfo !== undefined", timeout=60000)
            pg.evaluate("document.fonts.ready")
            return pg

        page = open_page()
        total = page.evaluate("window.directorInfo.frames")
        todo = [f for f in (frames if frames is not None else range(total))
                if frames is not None or not (out_dir / f"{f:05d}.jpg").exists()]     # resume: skip rendered frames
        t0 = time.time()
        for n, f in enumerate(todo):
            if n and n % 300 == 0:                                    # fresh page: keeps memory and tile cache bounded
                page.close(); page = open_page()
            for attempt in range(3):
                try:
                    page.evaluate(f"window.directorFrame({f})")
                    try:                                              # tiles for this camera position
                        page.wait_for_function("window.directorReady()", timeout=15000, polling=50)
                    except Exception:
                        pass
                    page.wait_for_timeout(15)
                    page.screenshot(path=str(out_dir / f"{f:05d}.jpg"), type="jpeg", quality=90, timeout=60000)
                    break
                except Exception as e:
                    print(f"frame {f}: {type(e).__name__}, reopening page (attempt {attempt + 1})", flush=True)
                    try:
                        page.close()
                    except Exception:
                        pass
                    page = open_page()
            if n % 100 == 0:
                print(f"frame {f}/{total}  {n}/{len(todo)} this run  {time.time() - t0:.0f}s", flush=True)
        browser.close()
    srv.shutdown()
    return total


def encode(frames_dir: Path, out: Path, fps: int = 30):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "%05d.jpg"),
                    "-vf", "fade=t=in:st=0:d=0.8,fade=t=out:st=76:d=1", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="residual_rl")
    ap.add_argument("--frames", help="comma-separated frame numbers for a preview (no video)")
    ap.add_argument("--out", default=str(VIEWER / "dubai_ai_taxi.mp4"))
    ap.add_argument("--fresh", action="store_true", help="discard frames from an earlier, interrupted render")
    a = ap.parse_args()
    fdir = VIEWER / ("preview_frames" if a.frames else "frames")
    if a.frames:
        render(a.method, [int(x) for x in a.frames.split(",")], fdir)
        print(f"preview frames in {fdir}")
    else:
        if a.fresh and fdir.exists():
            shutil.rmtree(fdir)
        render(a.method, None, fdir)
        encode(fdir, Path(a.out))
        print(f"video: {a.out}")

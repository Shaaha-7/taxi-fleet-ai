"""3D promo film: python -m src.simulation.make_video3d [--frames 300,1500] [--fresh]

Renders src/simulation/video3d.html (MapLibre: satellite imagery, 110k extruded OSM buildings, 3D taxis,
cinematic camera) frame by frame in headless Edge, then mixes the neural voice-over (outputs/video3d/vo/*.mp3,
placed by outputs/video3d/timeline.json) over the original score (outputs/video3d/music.wav), ducking the
music under the voice. Output: outputs/video3d/dubai_ai_taxi_3d.mp4 (+ _share.mp4).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from .make_video import ROOT, VIEWER, serve

V3 = ROOT / "outputs" / "video3d"


def render(frames: list[int] | None, out_dir: Path, port=8771, width=1920, height=1080, google: bool = False) -> int:
    from playwright.sync_api import sync_playwright
    page_name = "video3d_google.html" if google else "video3d.html"
    shutil.copy(ROOT / "src" / "simulation" / page_name, VIEWER / page_name)
    key_js = VIEWER / "gkey.js"
    if google:                                    # key is served to localhost only and removed after the render
        key_js.write_text("window.GKEY = " + json.dumps((ROOT / "secrets" / "google_maps.key").read_text().strip()) + ";\n")
    tl = json.loads((V3 / "timeline.json").read_text())
    (VIEWER / "timeline.js").write_text("window.TIMELINE = " + json.dumps(tl) + ";\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    srv = serve(port)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True,
                                    args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"])

        def open_page():
            pg = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            pg.goto(f"http://127.0.0.1:{port}/{page_name}", wait_until="networkidle")
            pg.wait_for_function("window.ready3d === true" if google else
                                 "window.ready3d === true && map.isSourceLoaded('bld') && map.isSourceLoaded('roads')", timeout=180000)
            pg.evaluate("document.fonts.ready")
            return pg

        page = open_page()
        total = page.evaluate("info3d().frames")
        todo = [f for f in (frames if frames is not None else range(total))
                if frames is not None or not (out_dir / f"{f:05d}.jpg").exists()]
        t0 = time.time()
        for n, f in enumerate(todo):
            if n and n % 400 == 0:
                page.close(); page = open_page()
            for attempt in range(3):
                try:
                    page.evaluate(f"frame3d({f})")
                    page.screenshot(path=str(out_dir / f"{f:05d}.jpg"), type="jpeg", quality=92, timeout=90000)
                    break
                except Exception as e:
                    print(f"frame {f}: {type(e).__name__} {str(e)[:120]}; reopening (attempt {attempt + 1})", flush=True)
                    try:
                        page.close()
                    except Exception:
                        pass
                    page = open_page()
            if n % 100 == 0:
                print(f"frame {f}/{total}  {n}/{len(todo)} this run  {time.time() - t0:.0f}s", flush=True)
        browser.close()
    srv.shutdown()
    key_js.unlink(missing_ok=True)
    return total


def mix(frames_dir: Path, out: Path, crf: int = 18, vo_dir: str = "vo"):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tl = json.loads((V3 / "timeline.json").read_text())
    total = max(s["end"] for s in tl)
    inputs, chains, labels = ["-framerate", "30", "-i", str(frames_dir / "%05d.jpg"), "-i", str(V3 / "music.wav")], [], []
    for k, s in enumerate(tl):
        inputs += ["-i", str(V3 / vo_dir / f"{s['name']}.mp3")]
        ms = int(s["vo_at"] * 1000)
        chains.append(f"[{k + 2}:a]aresample=44100,adelay={ms}|{ms},apad[v{k}]")
        labels.append(f"[v{k}]")
    fil = ";".join(chains) + f";{''.join(labels)}amix=inputs={len(tl)}:normalize=0,atrim=0:{total},volume=1.6[voice];" \
          "[voice]asplit=2[vo1][vo2];" \
          "[1:a]volume=0.55[mus];[mus][vo1]sidechaincompress=threshold=0.04:ratio=6:attack=40:release=500[duck];" \
          f"[duck][vo2]amix=inputs=2:normalize=0,alimiter=limit=0.95[a];" \
          f"[0:v]eq=contrast=1.05:saturation=1.12:gamma=0.98,noise=alls=3:allf=t," \
          f"fade=t=in:st=0:d=0.8,fade=t=out:st={total - 1.2:.2f}:d=1.2,format=yuv420p[v]"
    subprocess.run([ff, "-y", "-loglevel", "error", *inputs, "-filter_complex", fil, "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--mix-only", action="store_true")
    ap.add_argument("--vo", default="vo", help="voice-over folder in outputs/video3d (vo = edge-tts Andrew, vo_eleven = ElevenLabs)")
    ap.add_argument("--google", action="store_true", help="photoreal Google 3D Tiles version (video3d_google.html)")
    ap.add_argument("--tag", default="", help="suffix for the output file names, e.g. _brian")
    a = ap.parse_args()
    fdir = V3 / (("preview" if a.frames else "frames") + ("_google" if a.google else ""))
    if a.frames:
        render([int(x) for x in a.frames.split(",")], fdir, google=a.google)
        print("preview frames in", fdir)
    else:
        if a.fresh and fdir.exists():
            shutil.rmtree(fdir)
        if not a.mix_only:
            render(None, fdir, google=a.google)
        mix(fdir, V3 / f"dubai_ai_taxi_3d{a.tag}.mp4", 18, a.vo)
        mix(fdir, V3 / f"dubai_ai_taxi_3d{a.tag}_share.mp4", 25, a.vo)
        print("video:", V3 / f"dubai_ai_taxi_3d{a.tag}.mp4")

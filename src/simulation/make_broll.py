"""AI B-roll for the film's opening, generated locally with LTX-Video in ComfyUI, then the final cut.

    1. start C:\\ComfyUI_windows_portable\\run_nvidia_gpu.bat (serves http://127.0.0.1:8188)
    2. python -m src.simulation.make_broll generate          # 3 shots x N takes -> outputs/video3d/broll/
    3. python -m src.simulation.make_broll cut --take skyline=0,rider=1,driver=0 [--film photoreal]

The opening narration ("Dubai moves millions... a rider left waiting... a driver earning nothing") plays over the
three shots; the 3D film then continues from its title card. AI footage is illustrative only: it is not the
simulation and carries no data.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "outputs" / "video3d"
BROLL = V3 / "broll"
COMFY = "http://127.0.0.1:8188"
COMFY_OUT = Path(r"C:\ComfyUI_windows_portable\ComfyUI\output")

NEG = ("low quality, worst quality, deformed, distorted, disfigured, blurry, text, watermark, logo, subtitles, "
       "jittery, flicker, cartoon, illustration, oversaturated")
SHOTS = {
    "skyline": "Cinematic aerial drone shot slowly flying toward the Dubai skyline at blue hour, the Burj Khalifa glowing, "
               "skyscrapers with lit windows, long exposure light trails of traffic on a wide multi-lane highway below, "
               "warm sunset sky, smooth camera motion, ultra realistic, 4k film look",
    "rider":   "Cinematic medium shot at night in a modern city, a young businesswoman in a dark blazer stands at the edge of "
               "a sidewalk checking her smartphone and looking down the street for a taxi, soft bokeh of city lights and "
               "passing car headlights behind her, shallow depth of field, realistic, 35mm film look",
    "driver":  "Cinematic close-up inside a parked taxi at night, a male driver's hands resting on the steering wheel, "
               "a dashboard phone glowing with a navigation map, city lights reflecting on the windshield, rain-free, "
               "moody teal and orange color grade, shallow depth of field, realistic, 35mm film look",
}


def post(path, payload):
    req = urllib.request.Request(COMFY + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(COMFY + path, timeout=60) as r:
        return json.load(r)


def workflow(prompt: str, seed: int, prefix: str, width=768, height=512, frames=97, steps=8) -> dict:
    """ComfyUI API graph for LTX-Video text-to-video with core nodes (distilled model: few steps, cfg 1)."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltxv-2b-0.9.8-distilled-fp8.safetensors"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "ltxv"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["2", 0]}},
        "5": {"class_type": "LTXVConditioning", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": 24}},
        "6": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}},
        "7": {"class_type": "LTXVScheduler", "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": True,
                                                         "terminal": 0.1, "latent": ["6", 0]}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "SamplerCustom", "inputs": {"model": ["1", 0], "add_noise": True, "noise_seed": seed, "cfg": 1.0,
                                                         "positive": ["5", 0], "negative": ["5", 1], "sampler": ["8", 0],
                                                         "sigmas": ["7", 0], "latent_image": ["6", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def generate(takes: int):
    BROLL.mkdir(parents=True, exist_ok=True)
    get("/system_stats")                                                  # fails fast if ComfyUI is not running
    for name, prompt in SHOTS.items():
        for k in range(takes):
            prefix = f"broll_{name}_{k}"
            seed = random.Random(f"{name}{k}").randint(0, 2**31)
            t0 = time.time()
            pid = post("/prompt", {"prompt": workflow(prompt, seed, prefix)})["prompt_id"]
            while True:
                h = get(f"/history/{pid}")
                if pid in h and h[pid].get("status", {}).get("completed"):
                    break
                if pid in h and h[pid].get("status", {}).get("status_str") == "error":
                    raise RuntimeError(json.dumps(h[pid]["status"])[:800])
                time.sleep(3)
            imgs = [i for o in h[pid]["outputs"].values() for i in o.get("images", [])]
            dst = BROLL / f"{name}_{k}"
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir()
            for n, im in enumerate(imgs):
                shutil.copy(COMFY_OUT / im.get("subfolder", "") / im["filename"], dst / f"{n:04d}.png")
            mp4 = BROLL / f"{name}_{k}.mp4"
            encode_clip(dst, mp4)
            print(f"{name} take {k}: {len(imgs)} frames in {time.time() - t0:.0f}s -> {mp4.name}", flush=True)


def encode_clip(frames: Path, out: Path):
    """24 fps LTX frames -> 1080p30 clip: Lanczos upscale, crop to 16:9, light grade to match the film."""
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "24", "-i", str(frames / "%04d.png"),
                    "-vf", "scale=1920:1280:flags=lanczos,crop=1920:1080,fps=30,eq=contrast=1.04:saturation=1.08,"
                           "unsharp=5:5:0.6,noise=alls=3:allf=t,format=yuv420p",
                    "-c:v", "libx264", "-crf", "16", "-preset", "slow", str(out)], check=True)


def cut(takes: dict, film: str, vo_dir: str):
    """B-roll (with the intro line) + the film from its title card, voice re-timed, fresh score."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tl = json.loads((V3 / "timeline.json").read_text())
    intro_end = next(s["end"] for s in tl if s["name"] == "intro")
    per = intro_end / 3                                                 # each shot ~3.4 s
    skip = 5.0                                                           # film resumes at its title card
    shift = 3 * per - 0.2 - skip                                         # film time t plays at t + shift (0.6 s crossfade)
    total = max(s["end"] for s in tl) + shift
    frames = V3 / ("frames_google" if film == "photoreal" else "frames")
    from .video_music import render, save_wav
    results = next(s["start"] for s in tl if s["name"] == "results") + shift
    end_at = next(s["start"] for s in tl if s["name"] == "end") + shift
    music = V3 / "music_broll.wav"
    save_wav(render(total, intro_end + 4, results, end_at), str(music))
    clips = [BROLL / f"{n}_{takes.get(n, 0)}.mp4" for n in SHOTS]
    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]
    inputs += ["-framerate", "30", "-start_number", str(int(skip * 30)), "-i", str(frames / "%05d.jpg"), "-i", str(music)]
    vparts = []
    for k in range(3):                                                  # trim each shot, short crossfade into the next
        vparts.append(f"[{k}:v]trim=0:{per + 0.4:.2f},setpts=PTS-STARTPTS,fps=30,settb=1/30[b{k}]")
    vparts.append(f"[3:v]fps=30,settb=1/30,eq=contrast=1.05:saturation=1.12:gamma=0.98,noise=alls=3:allf=t[film]")
    vparts.append(f"[b0][b1]xfade=transition=fade:duration=0.4:offset={per:.2f}[x1]")
    vparts.append(f"[x1][b2]xfade=transition=fade:duration=0.4:offset={2 * per:.2f}[x2]")
    vparts.append(f"[x2][film]xfade=transition=fade:duration=0.6:offset={3 * per - 0.2:.2f},"
                  f"fade=t=in:st=0:d=0.8,fade=t=out:st={total - 1.2:.2f}:d=1.2,format=yuv420p[v]")
    achains, labels = [], []
    for k, s in enumerate(tl):
        inputs += ["-i", str(V3 / vo_dir / f"{s['name']}.mp3")]
        at = s["vo_at"] if s["name"] == "intro" else s["vo_at"] + shift
        ms = int(at * 1000)
        achains.append(f"[{k + 5}:a]aresample=44100,adelay={ms}|{ms},apad[v{k}]")
        labels.append(f"[v{k}]")
    afil = ";".join(achains) + f";{''.join(labels)}amix=inputs={len(tl)}:normalize=0,atrim=0:{total:.2f},volume=1.6[voice];" \
           "[voice]asplit=2[vo1][vo2];[4:a]volume=0.55[mus];[mus][vo1]sidechaincompress=threshold=0.04:ratio=6:attack=40:release=500[duck];" \
           "[duck][vo2]amix=inputs=2:normalize=0,alimiter=limit=0.95[a]"
    for crf, suffix in ((18, ""), (25, "_share")):
        out = V3 / f"dubai_ai_taxi_{film}_broll{suffix}.mp4"
        subprocess.run([ff, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(vparts) + ";" + afil,
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                        "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.2f}", "-movflags", "+faststart", str(out)], check=True)
        print("video:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["generate", "cut"])
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--take", default="", help="chosen take per shot, e.g. skyline=0,rider=1,driver=0")
    ap.add_argument("--film", default="photoreal", choices=["photoreal", "night"])
    ap.add_argument("--vo", default="vo_eleven")
    a = ap.parse_args()
    if a.action == "generate":
        generate(a.takes)
    else:
        cut({k: int(v) for k, v in (p.split("=") for p in a.take.split(",") if p)}, a.film, a.vo)

"""Original background score for the promo video, synthesised from scratch (no samples, no licence issues).

Corporate-tech mood at 100 BPM in A minor: warm detuned pads (Am - F - C - G), a sub bass, a plucked
16th-note arpeggio that enters after the intro, a soft four-on-the-floor pulse for the middle section,
a noise riser into the results, and a low hit with a long tail at the end card.
    python -m src.simulation.video_music --seconds 88 --out outputs/video3d/music.wav
"""
from __future__ import annotations

import argparse
import wave

import numpy as np

SR = 44100
BPM = 100
BEAT = 60 / BPM
CHORDS = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]      # Am F C G (MIDI)
hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)


def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):                       # one-pole; fine for a ~90 s mono buffer
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def env(n, attack, release):
    e = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    e[:a] = np.linspace(0, 1, a)
    e[n - r:] *= np.linspace(1, 0, r)
    return e


def render(seconds: float, intro_end: float, results_at: float, end_at: float) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    bar = 4 * BEAT
    rng = np.random.default_rng(7)
    # Pads: 2 bars per chord, three detuned saws per note, slow filter swell.
    pad = np.zeros(n)
    for k in range(int(seconds / (2 * bar)) + 1):
        s, e = int(k * 2 * bar * SR), min(int((k + 1) * 2 * bar * SR) + int(0.6 * SR), n)
        if s >= n:
            break
        tt = t[s:e] - t[s]
        chunk = np.zeros(e - s)
        for m in CHORDS[k % 4]:
            for det in (-0.08, 0.0, 0.07):
                f = hz(m - 12) * 2 ** (det / 12)
                chunk += 2 * (tt * f % 1) - 1
        pad[s:e] += chunk * env(e - s, 0.9, 0.9) / 9
    pad = lowpass(pad, 900)
    # Sub bass on chord roots.
    bass = np.zeros(n)
    for k in range(int(seconds / (2 * bar)) + 1):
        s, e = int(k * 2 * bar * SR), min(int((k + 1) * 2 * bar * SR), n)
        if s >= n:
            break
        tt = t[s:e] - t[s]
        bass[s:e] = np.sin(2 * np.pi * hz(CHORDS[k % 4][0] - 24) * tt) * env(e - s, 0.05, 0.3)
    # Plucked arpeggio, 16ths, after the intro.
    arp = np.zeros(n)
    step = BEAT / 4
    pattern = [0, 1, 2, 1, 2, 0, 1, 2]
    for j in range(int(intro_end / step), int(end_at / step)):
        s = int(j * step * SR)
        if s >= n:
            break
        chord = CHORDS[int(j * step / (2 * bar)) % 4]
        f = hz(chord[pattern[j % 8]] + 12)
        L = min(int(0.32 * SR), n - s)
        tt = np.arange(L) / SR
        arp[s:s + L] += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 14)
    # Soft kick pulse in the middle section.
    kick = np.zeros(n)
    for j in range(int(intro_end / BEAT) + 4, int(results_at / BEAT)):
        s = int(j * BEAT * SR)
        L = min(int(0.25 * SR), n - s)
        tt = np.arange(L) / SR
        kick[s:s + L] += np.sin(2 * np.pi * (48 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
    # Riser into the results, hit at the end card.
    riser = np.zeros(n)
    rs, re_ = int((results_at - 4) * SR), int(results_at * SR)
    riser[rs:re_] = lowpass(rng.standard_normal(re_ - rs), 2500) * np.linspace(0, 1, re_ - rs) ** 2
    hit = np.zeros(n)
    hs = int(end_at * SR)
    L = n - hs
    tt = np.arange(L) / SR
    hit[hs:] = (np.sin(2 * np.pi * 55 * tt) + 0.5 * np.sin(2 * np.pi * 110 * tt)) * np.exp(-tt * 0.9)
    mono = 0.55 * pad + 0.35 * bass + 0.16 * arp + 0.4 * kick + 0.12 * riser + 0.5 * hit
    width = lowpass(np.roll(pad, 441), 1200) * 0.25                   # simple stereo spread from the pad
    out[:, 0], out[:, 1] = mono + width, mono - width
    out *= env(n, 2.0, 3.0)[:, None]
    return out / np.abs(out).max() * 0.7


def save_wav(x: np.ndarray, path: str):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=88)
    ap.add_argument("--intro", type=float, default=9)
    ap.add_argument("--results", type=float, default=72)
    ap.add_argument("--end", type=float, default=80)
    ap.add_argument("--out", default="outputs/video3d/music.wav")
    a = ap.parse_args()
    save_wav(render(a.seconds, a.intro, a.results, a.end), a.out)
    print("wrote", a.out)

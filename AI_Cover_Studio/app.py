import os
import re
import sys
import html
import json
import asyncio
import random
import base64
import shutil
import zipfile
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pyworld as pw
import torch

# Maximize PyTorch CPU parallelism
try:
    _cpu_n = os.cpu_count() or 4
    torch.set_num_threads(_cpu_n)
except Exception:
    pass
import ffmpeg
import gradio as gr
from pydub import AudioSegment
from huggingface_hub import HfApi, hf_hub_download
from pedalboard import (
    Pedalboard,
    Reverb,
    Compressor,
    HighpassFilter,
    Limiter,
)

# ---------------------------------------------------------------------------
# Paths & Environment Configuration
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(APP_DIR, "output")
RVC_DIR = os.path.join(PROJECT_ROOT, "rvc_cli")
VENV_SCRIPTS = os.path.join(PROJECT_ROOT, "venv", "Scripts")
FFMPEG_DIR = os.path.join(
    PROJECT_ROOT, "venv", "Lib", "site-packages", "static_ffmpeg", "bin", "win32"
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.isdir(FFMPEG_DIR):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

FFMPEG_EXE = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
FFPROBE_EXE = os.path.join(FFMPEG_DIR, "ffprobe.exe")
if not os.path.exists(FFMPEG_EXE):
    FFMPEG_EXE = shutil.which("ffmpeg") or "ffmpeg"
if not os.path.exists(FFPROBE_EXE):
    FFPROBE_EXE = shutil.which("ffprobe") or "ffprobe"

AudioSegment.converter = FFMPEG_EXE
AudioSegment.ffprobe = FFPROBE_EXE

YT_DLP_EXE = os.path.join(VENV_SCRIPTS, "yt-dlp.exe")
DEMUCS_EXE = os.path.join(VENV_SCRIPTS, "demucs.exe")

# Global cache of scanned models: {label: {"pth": ..., "index": ...}}
MODEL_REGISTRY: dict[str, dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sanitize_name(name: str, default: str = "track") -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip()
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    return cleaned[:60] if cleaned else default


def scan_local_models() -> tuple[dict[str, dict[str, str]], list[str]]:
    """
    Scans ../models for all subdirectories containing .pth and .index files.
    Returns (registry_dict, sorted_labels).
    """
    registry: dict[str, dict[str, str]] = {}
    if not os.path.isdir(MODELS_DIR):
        return registry, []

    for root, dirs, files in os.walk(MODELS_DIR):
        # Skip cache or download archive folders
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and not d.endswith("_downloads")
        ]
        pth_files = sorted(
            f for f in files
            if f.lower().endswith(".pth") and not f.startswith(("G_", "D_"))
        )
        idx_files = sorted(
            f for f in files if f.lower().endswith(".index")
        )
        if pth_files and idx_files:
            rel = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
            label = rel if rel != "." else os.path.splitext(pth_files[0])[0]
            registry[label] = {
                "pth": os.path.join(root, pth_files[0]),
                "index": os.path.join(root, idx_files[0]),
            }
        elif pth_files and not idx_files:
            # Also support models that have .pth without .index
            rel = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
            label = f"{rel} (no index)" if rel != "." else f"{os.path.splitext(pth_files[0])[0]} (no index)"
            registry[label] = {
                "pth": os.path.join(root, pth_files[0]),
                "index": "",
            }

    labels = sorted(registry.keys(), key=str.lower)
    return registry, labels


def refresh_models_ui():
    global MODEL_REGISTRY
    MODEL_REGISTRY, labels = scan_local_models()
    default_1 = labels[0] if labels else None
    default_2 = labels[1] if len(labels) > 1 else default_1
    return (
        gr.update(choices=labels, value=default_1),
        gr.update(choices=labels, value=default_2),
    )


ARCHIVE_REGISTRY: dict[str, str] = {}


def scan_archive_files() -> tuple[dict[str, str], list[str]]:
    """
    Scans OUTPUT_DIR (and existing project cover folders) for generated .mp3 files,
    sorted by modification time (newest first).
    """
    found: list[tuple[float, str, str]] = []
    skip_dirs = {"venv", "rvc_cli", "models", "separated", "__pycache__", ".git", "assets"}

    for base_dir in (OUTPUT_DIR, PROJECT_ROOT):
        if not os.path.isdir(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for f in files:
                if f.lower().endswith(".mp3"):
                    full_p = os.path.abspath(os.path.join(root, f))
                    try:
                        mtime = os.path.getmtime(full_p)
                        size_mb = os.path.getsize(full_p) / (1024 * 1024)
                        dt_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                        rel = os.path.relpath(full_p, PROJECT_ROOT).replace("\\", "/")
                        label = f"{rel} ({size_mb:.1f} MB - {dt_str})"
                        found.append((mtime, label, full_p))
                    except OSError:
                        pass

    found.sort(key=lambda x: x[0], reverse=True)
    registry: dict[str, str] = {}
    labels: list[str] = []
    seen_paths = set()
    for _, label, full_p in found:
        if full_p in seen_paths:
            continue
        seen_paths.add(full_p)
        registry[label] = full_p
        labels.append(label)
    return registry, labels


def refresh_archive_ui():
    global ARCHIVE_REGISTRY
    ARCHIVE_REGISTRY, labels = scan_archive_files()
    first_val = labels[0] if labels else None
    first_path = ARCHIVE_REGISTRY.get(first_val) if first_val else None
    return (
        gr.update(choices=labels, value=first_val),
        first_path,
        first_path,
    )


def select_archive_file(label: str):
    path = ARCHIVE_REGISTRY.get(label) if label else None
    if path and os.path.exists(path):
        return path, path
    return None, None


def estimate_median_f0(wav_path: str) -> float:
    """Estimates the median F0 (Hz) of active voiced frames using pyworld."""
    y, sr = sf.read(wav_path, always_2d=True)
    y_mono = np.mean(y, axis=1).astype(np.float64)
    if len(y_mono) == 0:
        return 0.0

    frame_ms = 20.0
    _f0, t = pw.dio(y_mono, sr, frame_period=frame_ms, f0_floor=65.0, f0_ceil=600.0)
    f0 = pw.stonemask(y_mono, _f0, t, sr)

    hop = int(sr * frame_ms / 1000.0)
    n_frames = len(f0)
    rms = np.zeros(n_frames, dtype=np.float64)
    for idx in range(n_frames):
        s = idx * hop
        e = min(len(y_mono), s + hop)
        if e > s:
            rms[idx] = np.sqrt(np.mean(y_mono[s:e] ** 2))

    active = (f0 > 70.0) & (rms > 0.008)
    voiced = f0[active]
    return float(np.median(voiced)) if len(voiced) > 0 else 0.0


def decide_auto_pitch(wav_path: str, model_label: str) -> tuple[int, float]:
    """
    Estimates median F0 of the vocal track and chooses 0, +12, or -12
    based on vocal register and target voice hints.
    """
    med_f0 = estimate_median_f0(wav_path)
    if med_f0 <= 0.0:
        return 0, 0.0

    name_lower = model_label.lower()
    female_hints = ("miku", "teto", "female", "woman", "girl", "soprano", "rin", "luka", "gumi")
    male_hints = ("pandetta", "male", "man", "boy", "kaito", "gakupo", "baritone", "bass")

    is_female_target = any(h in name_lower for h in female_hints)
    is_male_target = any(h in name_lower for h in male_hints)

    if is_female_target and not is_male_target:
        # Target is high/female register (~220-260 Hz)
        pitch = 12 if med_f0 < 175.0 else 0
    elif is_male_target and not is_female_target:
        # Target is low/male register (~115-155 Hz)
        pitch = -12 if med_f0 > 185.0 else 0
    else:
        # Neutral register heuristic
        if med_f0 < 160.0:
            pitch = 12
        elif med_f0 > 245.0:
            pitch = -12
        else:
            pitch = 0

    return pitch, med_f0


def resolve_pitch(mode: str, manual_val: float, wav_path: str, model_label: str) -> tuple[int, str]:
    if mode == "Auto":
        pitch, med_f0 = decide_auto_pitch(wav_path, model_label)
        return pitch, f"Auto (F0 mediana = {med_f0:.1f} Hz -> pitch={pitch:+d})"
    if mode == "Manuale":
        p = int(round(manual_val))
        return p, f"Manuale ({p:+d})"
    p = int(mode)
    return p, f"Fisso ({p:+d})"


def load_stereo(path: str, target_sr: int = 44100) -> np.ndarray:
    y, sr = sf.read(path, always_2d=True)
    y = y.T.astype(np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    elif y.shape[0] > 2:
        y = y[:2, :]
    return y


def split_vocals_by_f0(
    vocals_path: str,
    male_out: str,
    female_out: str,
    fade_ms: int = 100,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Separates vocals.wav into complementary male and female tracks
    using pyworld F0 analysis + pydub with 100ms crossfade.
    """
    y, sr = sf.read(vocals_path, always_2d=True)
    y_mono = np.mean(y, axis=1).astype(np.float64)

    frame_ms = 20.0
    _f0, t = pw.dio(y_mono, sr, frame_period=frame_ms, f0_floor=65.0, f0_ceil=600.0)
    f0 = pw.stonemask(y_mono, _f0, t, sr)

    hop = int(sr * frame_ms / 1000.0)
    n_frames = len(f0)
    rms = np.zeros(n_frames, dtype=np.float64)
    for idx in range(n_frames):
        s = idx * hop
        e = min(len(y_mono), s + hop)
        if e > s:
            rms[idx] = np.sqrt(np.mean(y_mono[s:e] ** 2))

    active_voiced = (f0 > 70.0) & (rms > 0.008)
    all_voiced_f0 = f0[active_voiced]
    global_med = float(np.median(all_voiced_f0)) if len(all_voiced_f0) > 0 else 145.0

    win_frames = 12
    step_frames = 6
    female_threshold_hz = max(210.0, global_med * 1.40)

    female_frame_mask = np.zeros(n_frames, dtype=bool)
    for start_f in range(0, max(0, n_frames - win_frames + 1), step_frames):
        end_f = start_f + win_frames
        seg_mask = active_voiced[start_f:end_f]
        if np.sum(seg_mask) >= 4:
            med_f0 = float(np.median(f0[start_f:end_f][seg_mask]))
            if med_f0 >= female_threshold_hz:
                female_frame_mask[start_f:end_f] = True

    raw_intervals = []
    i = 0
    while i < n_frames:
        if not female_frame_mask[i]:
            i += 1
            continue
        start_f = i
        while i < n_frames and female_frame_mask[i]:
            i += 1
        end_f = i
        raw_intervals.append([int(start_f * frame_ms), int(end_f * frame_ms)])

    merged = []
    for s_ms, e_ms in raw_intervals:
        if not merged or s_ms - merged[-1][1] > 450:
            merged.append([s_ms, e_ms])
        else:
            merged[-1][1] = e_ms

    vocals_seg = AudioSegment.from_wav(vocals_path)
    total_len = len(vocals_seg)

    female_intervals = []
    female_f0_values = []
    for s_ms, e_ms in merged:
        if (e_ms - s_ms) >= 300:
            ps = max(0, s_ms - 120)
            pe = min(total_len, e_ms + 120)
            female_intervals.append((ps, pe))
            f_mask = (t * 1000.0 >= ps) & (t * 1000.0 <= pe) & active_voiced
            if np.any(f_mask):
                female_f0_values.extend(f0[f_mask].tolist())

    final_female = []
    for s_ms, e_ms in female_intervals:
        if not final_female or s_ms > final_female[-1][1]:
            final_female.append([s_ms, e_ms])
        else:
            final_female[-1][1] = max(final_female[-1][1], e_ms)
    final_female_tuples = [(s, e) for s, e in final_female]

    female_med_hz = float(np.median(female_f0_values)) if female_f0_values else 0.0

    male_timeline = AudioSegment.empty()
    female_timeline = AudioSegment.empty()
    cursor = 0

    for s_ms, e_ms in final_female_tuples:
        if s_ms > cursor:
            seg_m = vocals_seg[cursor:s_ms]
            if cursor > 0 and len(seg_m) > 0:
                seg_m = seg_m.fade_in(min(fade_ms, len(seg_m)))
            if len(seg_m) > 0:
                seg_m = seg_m.fade_out(min(fade_ms, len(seg_m)))
            male_timeline += seg_m

            female_timeline += AudioSegment.silent(
                duration=s_ms - cursor,
                frame_rate=vocals_seg.frame_rate,
            ).set_channels(vocals_seg.channels).set_sample_width(vocals_seg.sample_width)

        seg_f = vocals_seg[s_ms:e_ms]
        if len(seg_f) > 0:
            seg_f = seg_f.fade_in(min(fade_ms, len(seg_f))).fade_out(min(fade_ms, len(seg_f)))
        female_timeline += seg_f

        male_timeline += AudioSegment.silent(
            duration=e_ms - s_ms,
            frame_rate=vocals_seg.frame_rate,
        ).set_channels(vocals_seg.channels).set_sample_width(vocals_seg.sample_width)
        cursor = e_ms

    if cursor < total_len:
        seg_m = vocals_seg[cursor:total_len]
        if cursor > 0 and len(seg_m) > 0:
            seg_m = seg_m.fade_in(min(fade_ms, len(seg_m)))
        male_timeline += seg_m

        female_timeline += AudioSegment.silent(
            duration=total_len - cursor,
            frame_rate=vocals_seg.frame_rate,
        ).set_channels(vocals_seg.channels).set_sample_width(vocals_seg.sample_width)

    male_timeline.export(male_out, format="wav")
    female_timeline.export(female_out, format="wav")

    return final_female_tuples, global_med, female_med_hz


def run_rvc_inference(
    input_wav: str,
    output_wav: str,
    pth_path: str,
    index_path: str,
    pitch: int,
    index_rate: float,
):
    prev_cwd = os.getcwd()
    try:
        os.chdir(RVC_DIR)
        if RVC_DIR not in sys.path:
            sys.path.insert(0, RVC_DIR)
        from core import run_infer_script

        msg, out_file = run_infer_script(
            pitch=pitch,
            index_rate=float(index_rate),
            volume_envelope=1.0,
            protect=0.33,
            f0_method="rmvpe",
            input_path=input_wav,
            output_path=output_wav,
            pth_path=pth_path,
            index_path=index_path or "",
            f0_autotune=False,
            f0_autotune_strength=1.0,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
            export_format="WAV",
            embedder_model="contentvec",
        )
        return msg, out_file
    finally:
        os.chdir(prev_cwd)


def run_rvc_active_only(
    input_wav: str,
    output_wav: str,
    pth_path: str,
    index_path: str,
    pitch: int,
    index_rate: float,
):
    """
    Runs RVC inference only on non-silent audio regions (merging gaps <= 1.2s)
    and places the converted segments back into a full-length silent timeline
    at their exact timestamps. Avoids wasting CPU time on silence.
    """
    seg = AudioSegment.from_wav(input_wav)
    total_ms = len(seg)
    if total_ms == 0:
        seg.export(output_wav, format="wav")
        return "Empty audio", output_wav

    y, sr = sf.read(input_wav, always_2d=True)
    y_mono = np.mean(y, axis=1)
    win = max(1, int(sr * 0.05))  # 50ms windows
    n_wins = len(y_mono) // win
    if n_wins == 0:
        return run_rvc_inference(input_wav, output_wav, pth_path, index_path, pitch, index_rate)

    rms = np.sqrt(np.mean(y_mono[: n_wins * win].reshape(n_wins, win) ** 2, axis=1))
    active = rms > 0.0015
    if not np.any(active):
        seg.export(output_wav, format="wav")
        return "Silent track skipped", output_wav

    raw_intervals = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        s = i
        while i < len(active) and active[i]:
            i += 1
        s_ms = max(0, s * 50 - 250)
        e_ms = min(total_ms, i * 50 + 250)
        raw_intervals.append([s_ms, e_ms])

    merged = []
    for s_ms, e_ms in raw_intervals:
        if not merged or s_ms - merged[-1][1] > 1200:
            merged.append([s_ms, e_ms])
        else:
            merged[-1][1] = max(merged[-1][1], e_ms)

    active_ms = sum(e - s for s, e in merged)
    # If >85% of the track is active, run directly on the whole file
    if active_ms >= total_ms * 0.85:
        return run_rvc_inference(input_wav, output_wav, pth_path, index_path, pitch, index_rate)

    base_dir = os.path.dirname(output_wav)
    timeline = AudioSegment.silent(
        duration=total_ms,
        frame_rate=seg.frame_rate,
    ).set_channels(seg.channels).set_sample_width(seg.sample_width)

    last_msg = "OK"
    for idx, (s_ms, e_ms) in enumerate(merged):
        tmp_in = os.path.join(base_dir, f"_tmp_chunk_in_{idx}.wav")
        tmp_out = os.path.join(base_dir, f"_tmp_chunk_out_{idx}.wav")
        try:
            seg[s_ms:e_ms].export(tmp_in, format="wav")
            last_msg, _ = run_rvc_inference(tmp_in, tmp_out, pth_path, index_path, pitch, index_rate)
            if os.path.exists(tmp_out):
                conv = AudioSegment.from_wav(tmp_out)
                if len(conv) > 40:
                    conv = conv.fade_in(20).fade_out(20)
                timeline = timeline.overlay(conv, position=s_ms)
        finally:
            for p in (tmp_in, tmp_out):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    timeline.export(output_wav, format="wav")
    return f"{last_msg} (active {active_ms/1000:.1f}s / {total_ms/1000:.1f}s)", output_wav


# ---------------------------------------------------------------------------
# Multi-Source RVC Model Search & Two-Step Download (HF Models/Datasets, GitHub, Direct URL)
# ---------------------------------------------------------------------------
def _github_get_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AI-Cover-Studio",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_url_to_file(url: str, dst_path: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst_path, "wb") as f:
        shutil.copyfileobj(resp, f)


def search_rvc_models(character_name: str, progress=gr.Progress()):
    logs = []

    def log(msg: str):
        logs.append(msg)
        return "\n".join(logs)

    query_raw = (character_name or "").strip()
    if not query_raw:
        yield (
            log("[ERRORE] Inserisci il nome di un personaggio o artista."),
            {},
            gr.update(choices=[], value=None, visible=False),
            gr.update(visible=False),
        )
        return

    progress(0.05, desc="Ricerca su Hugging Face e GitHub...")
    yield (
        log(f"[CERCA] Ricerca multi-fonte per '{query_raw}' (HF Models, HF Datasets, GitHub)..."),
        {},
        gr.update(choices=[], value=None, visible=False),
        gr.update(visible=False),
    )

    api = HfApi()
    queries = [query_raw, f"{query_raw} rvc"]
    candidates: dict[str, dict] = {}
    seen_keys = set()

    # 1. Hugging Face Models & Datasets
    for idx_q, q in enumerate(queries):
        progress(0.15 + 0.25 * idx_q, desc=f"Hugging Face Models: {q}...")
        try:
            for m in api.list_models(search=q, sort="downloads", limit=15):
                key = ("hf_model", m.id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                try:
                    files = api.list_repo_files(repo_id=m.id, repo_type="model")
                    pth_f = [
                        f for f in files
                        if f.lower().endswith(".pth") and not os.path.basename(f).startswith(("G_", "D_"))
                    ]
                    idx_f = [f for f in files if f.lower().endswith(".index")]
                    zip_f = [f for f in files if f.lower().endswith(".zip")]
                    if pth_f or zip_f:
                        dl = getattr(m, "downloads", 0) or 0
                        kind = ".pth + .index" if (pth_f and idx_f) else (".zip" if zip_f else ".pth")
                        label = f"[HF Model] {m.id} (dl: {dl} | {kind})"
                        candidates[label] = {
                            "source": "hf_model",
                            "repo_id": m.id,
                            "pth_files": pth_f,
                            "idx_files": idx_f,
                            "zip_files": zip_f,
                            "score": dl + (10000 if (pth_f and idx_f) else 0),
                        }
                except Exception:
                    pass
        except Exception as e:
            yield log(f"[AVVISO HF Model] {e}"), candidates, gr.update(), gr.update()

        progress(0.30 + 0.20 * idx_q, desc=f"Hugging Face Datasets: {q}...")
        try:
            for d in api.list_datasets(search=q, sort="downloads", limit=10):
                key = ("hf_dataset", d.id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                try:
                    files = api.list_repo_files(repo_id=d.id, repo_type="dataset")
                    pth_f = [
                        f for f in files
                        if f.lower().endswith(".pth") and not os.path.basename(f).startswith(("G_", "D_"))
                    ]
                    idx_f = [f for f in files if f.lower().endswith(".index")]
                    zip_f = [f for f in files if f.lower().endswith(".zip")]
                    if pth_f or zip_f:
                        dl = getattr(d, "downloads", 0) or 0
                        kind = ".pth + .index" if (pth_f and idx_f) else (".zip" if zip_f else ".pth")
                        label = f"[HF Dataset] {d.id} (dl: {dl} | {kind})"
                        candidates[label] = {
                            "source": "hf_dataset",
                            "repo_id": d.id,
                            "pth_files": pth_f,
                            "idx_files": idx_f,
                            "zip_files": zip_f,
                            "score": dl + (10000 if (pth_f and idx_f) else 0),
                        }
                except Exception:
                    pass
        except Exception as e:
            yield log(f"[AVVISO HF Dataset] {e}"), candidates, gr.update(), gr.update()

    # 2. GitHub Releases & Repositories
    progress(0.75, desc="Ricerca su GitHub...")
    try:
        gh_q = urllib.parse.quote(f"{query_raw} rvc")
        gh_res = _github_get_json(
            f"https://api.github.com/search/repositories?q={gh_q}&sort=stars&per_page=8"
        )
        for repo in gh_res.get("items", []):
            full_name = repo.get("full_name", "")
            stars = repo.get("stargazers_count", 0) or 0
            if not full_name:
                continue
            try:
                rels = _github_get_json(f"https://api.github.com/repos/{full_name}/releases?per_page=3")
                assets = []
                for rel in rels:
                    for a in rel.get("assets", []):
                        aname = a.get("name", "").lower()
                        if aname.endswith((".zip", ".pth", ".index")):
                            assets.append({"name": a["name"], "url": a["browser_download_url"]})
                if any(a["name"].lower().endswith((".zip", ".pth")) for a in assets):
                    label = f"[GitHub Release] {full_name} (stars: {stars} | {len(assets)} file)"
                    candidates[label] = {
                        "source": "github_release",
                        "repo_id": full_name,
                        "assets": assets,
                        "score": stars,
                    }
            except Exception:
                pass
    except Exception as e:
        yield log(f"[AVVISO GitHub] {e}"), candidates, gr.update(), gr.update()

    # 3. Voice-Models.com (uses POST search + page)
    progress(0.88, desc="Ricerca su Voice-Models.com...")
    try:
        post_data = urllib.parse.urlencode({"search": query_raw, "page": 1}).encode("utf-8")
        req = urllib.request.Request(
            "https://voice-models.com/fetch_data.php",
            data=post_data,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            vm_data = json.loads(resp.read().decode("utf-8", errors="replace"))
        tbl = vm_data.get("table", "")
        vm_count = 0
        for row in tbl.split("<tr>")[1:]:
            t_m = re.search(r"class=['\"]fs-5['\"]>([^<]+)</a>", row)
            u_m = re.search(r"easyaivoice\.com/run\?url=([^'\"&\s]+)", row)
            if not (t_m and u_m):
                continue
            title = html.unescape(t_m.group(1)).strip()
            dl_url = urllib.parse.unquote(u_m.group(1))
            if "huggingface.co/" in dl_url and "/blob/" in dl_url:
                dl_url = dl_url.replace("/blob/", "/resolve/")
            if "drive.google.com" in dl_url:
                continue
            if not dl_url.startswith(("http://", "https://")):
                continue
            label = f"[Voice-Models] {title}"
            if label not in candidates:
                candidates[label] = {
                    "source": "voice_models",
                    "repo_id": title,
                    "url": dl_url,
                    "score": 0,
                }
                vm_count += 1
            if vm_count >= 15:
                break
    except Exception as e:
        yield log(f"[AVVISO Voice-Models] {e}"), candidates, gr.update(), gr.update()

    # Filter & rank all candidates by query token relevance
    query_tokens = [w for w in re.split(r"[^a-z0-9]+", query_raw.lower()) if len(w) >= 2]
    if query_tokens:
        filtered: dict[str, dict] = {}
        for lbl, info in candidates.items():
            haystack = f"{info.get('repo_id', '')} {lbl}".lower()
            hits = sum(1 for tok in query_tokens if tok in haystack)
            if hits > 0:
                all_match = 100000 if hits == len(query_tokens) else 0
                info["score"] = (hits * 1_000_000) + all_match + info.get("score", 0)
                filtered[lbl] = info
        candidates = filtered

    sorted_labels = sorted(
        candidates.keys(),
        key=lambda k: candidates[k].get("score", 0),
        reverse=True,
    )

    if not sorted_labels:
        yield (
            log(f"[ERRORE] Nessun modello trovato su HF o GitHub per '{query_raw}'. Puoi usare l'URL diretto qui sotto."),
            {},
            gr.update(choices=[], value=None, visible=False),
            gr.update(visible=False),
        )
        return

    progress(1.0, desc="Ricerca completata!")
    yield (
        log(f"[TROVATI {len(sorted_labels)} MODELLI] Seleziona quello desiderato dal menu qui sotto e premi 'Conferma e Scarica'."),
        candidates,
        gr.update(choices=sorted_labels, value=sorted_labels[0], visible=True),
        gr.update(visible=True),
    )


def _finalize_downloaded_folder(repo_dir: str):
    """Finds .pth and .index in repo_dir and validates RVC v2 checkpoint."""
    local_pth = None
    local_idx = None
    for r, _, fnames in os.walk(repo_dir):
        for fn in sorted(fnames):
            if fn.lower().endswith(".pth") and not fn.startswith(("G_", "D_")) and not local_pth:
                local_pth = os.path.join(r, fn)
            elif fn.lower().endswith(".index") and not local_idx:
                local_idx = os.path.join(r, fn)

    if not local_pth or not os.path.exists(local_pth):
        return False, "Nessun file .pth valido trovato.", None

    ckpt = torch.load(local_pth, map_location="cpu", weights_only=False)
    version = ckpt.get("version")
    f0 = ckpt.get("f0")
    if version != "v2":
        return False, f"Checkpoint non RVC v2 (version={version}, f0={f0}).", None

    rel_label = os.path.relpath(os.path.dirname(local_pth), MODELS_DIR).replace("\\", "/")
    idx_note = "con .index" if (local_idx and os.path.exists(local_idx)) else "senza .index"
    return True, f"RVC v2 verificato ({idx_note}, f0={f0})!", rel_label


def confirm_download_model(
    selected_label: str,
    candidates_state: dict,
    character_name: str,
    progress=gr.Progress(),
):
    logs = []

    def log(msg: str):
        logs.append(msg)
        return "\n".join(logs)

    if not selected_label or not candidates_state or selected_label not in candidates_state:
        yield log("[ERRORE] Seleziona prima un modello dalla lista."), gr.update(), gr.update()
        return

    info = candidates_state[selected_label]
    source = info["source"]
    repo_id = info["repo_id"]
    safe_folder = sanitize_name(character_name or repo_id.split("/")[-1], default="custom_voice")
    repo_dir = os.path.join(MODELS_DIR, safe_folder, sanitize_name(repo_id.replace("/", "_")))
    os.makedirs(repo_dir, exist_ok=True)

    progress(0.15, desc=f"Download {repo_id}...")
    yield log(f"[DOWNLOAD] Scaricamento da {source}: {repo_id}..."), gr.update(), gr.update()

    try:
        if source in ("hf_model", "hf_dataset"):
            repo_type = "model" if source == "hf_model" else "dataset"
            pth_files = info.get("pth_files", [])
            idx_files = info.get("idx_files", [])
            zip_files = info.get("zip_files", [])
            if pth_files:
                hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=pth_files[0], local_dir=repo_dir)
                if idx_files:
                    hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=idx_files[0], local_dir=repo_dir)
            elif zip_files:
                local_zip = hf_hub_download(
                    repo_id=repo_id, repo_type=repo_type, filename=zip_files[0], local_dir=repo_dir
                )
                with zipfile.ZipFile(local_zip, "r") as zf:
                    zf.extractall(repo_dir)
        elif source == "github_release":
            assets = info.get("assets", [])
            zips = [a for a in assets if a["name"].lower().endswith(".zip")]
            pths = [a for a in assets if a["name"].lower().endswith(".pth")]
            idxs = [a for a in assets if a["name"].lower().endswith(".index")]
            if zips:
                dst = os.path.join(repo_dir, zips[0]["name"])
                _download_url_to_file(zips[0]["url"], dst)
                with zipfile.ZipFile(dst, "r") as zf:
                    zf.extractall(repo_dir)
            elif pths:
                _download_url_to_file(pths[0]["url"], os.path.join(repo_dir, pths[0]["name"]))
                if idxs:
                    _download_url_to_file(idxs[0]["url"], os.path.join(repo_dir, idxs[0]["name"]))
        elif source == "voice_models":
            dl_url = info["url"]
            fname = os.path.basename(urllib.parse.urlparse(dl_url).path) or "model.zip"
            if not fname.lower().endswith((".zip", ".pth")):
                fname += ".zip"
            dst = os.path.join(repo_dir, fname)
            _download_url_to_file(dl_url, dst)
            if dst.lower().endswith(".zip"):
                with zipfile.ZipFile(dst, "r") as zf:
                    zf.extractall(repo_dir)
    except Exception as e:
        shutil.rmtree(repo_dir, ignore_errors=True)
        yield log(f"[ERRORE DOWNLOAD] {e}"), gr.update(), gr.update()
        return

    progress(0.85, desc="Verifica RVC v2...")
    ok, msg, rel_label = _finalize_downloaded_folder(repo_dir)
    if not ok:
        shutil.rmtree(repo_dir, ignore_errors=True)
        yield log(f"[ERRORE VERIFICA] {msg}"), gr.update(), gr.update()
        return

    global MODEL_REGISTRY
    MODEL_REGISTRY, labels = scan_local_models()
    progress(1.0, desc="Completato!")
    yield (
        log(f"[COMPLETATO] {msg} Aggiunto come '{rel_label}'. Menu aggiornati!"),
        gr.update(choices=labels, value=rel_label),
        gr.update(choices=labels, value=rel_label),
    )


def install_from_direct_url(direct_url: str, custom_name: str, progress=gr.Progress()):
    logs = []

    def log(msg: str):
        logs.append(msg)
        return "\n".join(logs)

    url = (direct_url or "").strip()
    if not url.startswith(("http://", "https://")):
        yield log("[ERRORE] Inserisci un URL valido (http:// o https://)."), gr.update(), gr.update()
        return

    folder_name = sanitize_name(custom_name or "direct_voice", default="direct_voice")
    repo_dir = os.path.join(MODELS_DIR, folder_name)
    os.makedirs(repo_dir, exist_ok=True)

    filename = os.path.basename(urllib.parse.urlparse(url).path) or "model.zip"
    if not filename.lower().endswith((".zip", ".pth")):
        filename += ".zip"
    dst = os.path.join(repo_dir, filename)

    progress(0.2, desc="Download da URL diretto...")
    yield log(f"[URL] Scaricamento da {url}..."), gr.update(), gr.update()
    try:
        _download_url_to_file(url, dst)
        if dst.lower().endswith(".zip"):
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(repo_dir)
    except Exception as e:
        shutil.rmtree(repo_dir, ignore_errors=True)
        yield log(f"[ERRORE] Download fallito: {e}"), gr.update(), gr.update()
        return

    progress(0.85, desc="Verifica RVC v2...")
    ok, msg, rel_label = _finalize_downloaded_folder(repo_dir)
    if not ok:
        shutil.rmtree(repo_dir, ignore_errors=True)
        yield log(f"[ERRORE VERIFICA] {msg}"), gr.update(), gr.update()
        return

    global MODEL_REGISTRY
    MODEL_REGISTRY, labels = scan_local_models()
    progress(1.0, desc="Completato!")
    yield (
        log(f"[COMPLETATO] {msg} Aggiunto come '{rel_label}'. Menu aggiornati!"),
        gr.update(choices=labels, value=rel_label),
        gr.update(choices=labels, value=rel_label),
    )


# ---------------------------------------------------------------------------
# RVC v2 Voice Training Pipeline (via rvc_cli)
# ---------------------------------------------------------------------------
def train_rvc_voice(
    model_name: str,
    dataset_files: list[str] | None,
    extract_vocals_demucs: bool,
    sample_rate_str: str,
    total_epochs: int,
    batch_size: int,
    save_every_epoch: int,
    progress=gr.Progress(),
):
    logs = []

    def log(msg: str):
        logs.append(msg)
        return "\n".join(logs)

    safe_name = sanitize_name((model_name or "").strip(), default="")
    if not safe_name:
        yield log("[ERRORE] Inserisci un nome valido per la nuova voce."), gr.update(), gr.update()
        return

    if not dataset_files:
        yield log("[ERRORE] Carica almeno un file audio/video per il dataset."), gr.update(), gr.update()
        return

    sr = int(sample_rate_str)
    epochs = int(total_epochs)
    bs = int(batch_size)
    save_every = min(epochs, max(1, int(save_every_epoch)))
    cpu_n = os.cpu_count() or 4
    gpu_arg = 0 if torch.cuda.is_available() else "-"

    datasets_root = os.path.join(APP_DIR, "datasets", safe_name)
    raw_dir = os.path.join(datasets_root, "raw")
    clean_dir = os.path.join(datasets_root, "vocals")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    progress(0.05, desc="Preparazione dataset...")
    yield log(f"[TRAIN 1/5] Preparazione dataset per '{safe_name}' (device={'GPU' if gpu_arg == 0 else 'CPU'})..."), gr.update(), gr.update()

    converted_wavs = []
    for idx, fpath in enumerate(dataset_files):
        if not fpath or not os.path.exists(fpath):
            continue
        out_wav = os.path.join(raw_dir, f"clip_{idx:03d}.wav")
        try:
            (
                ffmpeg
                .input(fpath)
                .output(out_wav, acodec="pcm_s16le", ar=sr, ac=1)
                .overwrite_output()
                .run(cmd=FFMPEG_EXE, quiet=True)
            )
            if os.path.exists(out_wav) and os.path.getsize(out_wav) > 1024:
                converted_wavs.append(out_wav)
        except Exception as e:
            yield log(f"[AVVISO] Impossibile convertire {os.path.basename(fpath)}: {e}"), gr.update(), gr.update()

    if not converted_wavs:
        yield log("[ERRORE] Nessun file audio valido convertito per il training."), gr.update(), gr.update()
        return

    if extract_vocals_demucs:
        progress(0.15, desc="Estrazione voce con Demucs...")
        yield log(f"[TRAIN 1B/5] Separazione voce con Demucs su {len(converted_wavs)} file..."), gr.update(), gr.update()
        sep_dir = os.path.join(datasets_root, "separated")
        subprocess.run(
            [DEMUCS_EXE, "--two-stems=vocals", "-n", "htdemucs", "-j", str(cpu_n), "-o", sep_dir, *converted_wavs],
            check=True,
        )
        for idx, wpath in enumerate(converted_wavs):
            stem = Path(wpath).stem
            v_src = os.path.join(sep_dir, "htdemucs", stem, "vocals.wav")
            if os.path.exists(v_src):
                shutil.copy2(v_src, os.path.join(clean_dir, f"vocal_{idx:03d}.wav"))
    else:
        for idx, wpath in enumerate(converted_wavs):
            shutil.copy2(wpath, os.path.join(clean_dir, f"vocal_{idx:03d}.wav"))

    prev_cwd = os.getcwd()
    try:
        os.chdir(RVC_DIR)
        if RVC_DIR not in sys.path:
            sys.path.insert(0, RVC_DIR)
        import core as rvc_core

        progress(0.25, desc="Verifica prerequisiti HiFi-GAN...")
        yield log("[TRAIN 2/5] Verifica prerequisiti RVC v2 (HiFi-GAN pretrained)..."), gr.update(), gr.update()
        rvc_core.run_prerequisites_script(pretraineds_hifigan=True, models=False, exe=False)

        progress(0.35, desc="Preprocessing e taglio clip...")
        yield log("[TRAIN 3/5] Preprocessing dataset (taglio automatico)..."), gr.update(), gr.update()
        msg_prep = rvc_core.run_preprocess_script(
            model_name=safe_name,
            dataset_path=clean_dir,
            sample_rate=sr,
            cpu_cores=cpu_n,
            cut_preprocess="Automatic",
            process_effects=False,
            chunk_len=3.0,
            overlap_len=0.3,
            normalization_mode="none",
        )
        yield log(f"  -> {msg_prep}"), gr.update(), gr.update()

        progress(0.50, desc="Estrazione F0 (rmvpe) e feature (contentvec)...")
        yield log("[TRAIN 4/5] Estrazione pitch (rmvpe) e feature (contentvec)..."), gr.update(), gr.update()
        msg_ext = rvc_core.run_extract_script(
            model_name=safe_name,
            f0_method="rmvpe",
            cpu_cores=cpu_n,
            gpu=gpu_arg,
            sample_rate=sr,
            embedder_model="contentvec",
            include_mutes=2,
        )
        yield log(f"  -> {msg_ext}"), gr.update(), gr.update()

        progress(0.65, desc=f"Training modello ({epochs} epoche)...")
        yield log(f"[TRAIN 5/5] Avvio training RVC v2 ({epochs} epoche, batch={bs}, sr={sr}). Attendere..."), gr.update(), gr.update()
        msg_tr = rvc_core.run_train_script(
            model_name=safe_name,
            save_every_epoch=save_every,
            save_only_latest=True,
            save_every_weights=True,
            total_epoch=epochs,
            sample_rate=sr,
            batch_size=bs,
            gpu=gpu_arg,
            pretrained=True,
            cleanup=False,
            index_algorithm="Auto",
        )
        yield log(f"  -> {msg_tr}"), gr.update(), gr.update()
    except Exception as e:
        yield log(f"[ERRORE TRAINING] {e}"), gr.update(), gr.update()
        return
    finally:
        os.chdir(prev_cwd)

    # Locate generated .pth and .index in rvc_cli/logs/<safe_name>/
    rvc_logs_dir = os.path.join(RVC_DIR, "logs", safe_name)
    pth_candidates = []
    idx_candidates = []
    if os.path.isdir(rvc_logs_dir):
        for r, _, fnames in os.walk(rvc_logs_dir):
            for fn in fnames:
                fp = os.path.join(r, fn)
                if fn.lower().endswith(".pth") and not fn.startswith(("G_", "D_")):
                    pth_candidates.append(fp)
                elif fn.lower().endswith(".index") and "trained" not in fn.lower():
                    idx_candidates.append(fp)
        if not idx_candidates:
            for r, _, fnames in os.walk(rvc_logs_dir):
                for fn in fnames:
                    if fn.lower().endswith(".index"):
                        idx_candidates.append(os.path.join(r, fn))

    if not pth_candidates:
        yield log(f"[ERRORE] Training terminato ma nessun file .pth trovato in {rvc_logs_dir}."), gr.update(), gr.update()
        return

    # Pick newest / highest epoch .pth
    best_pth = max(pth_candidates, key=os.path.getmtime)
    dest_dir = os.path.join(MODELS_DIR, "custom", safe_name)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(best_pth, os.path.join(dest_dir, f"{safe_name}.pth"))
    if idx_candidates:
        best_idx = max(idx_candidates, key=os.path.getmtime)
        shutil.copy2(best_idx, os.path.join(dest_dir, f"{safe_name}.index"))

    global MODEL_REGISTRY
    MODEL_REGISTRY, labels = scan_local_models()
    new_label = f"custom/{safe_name}" if f"custom/{safe_name}" in MODEL_REGISTRY else (labels[0] if labels else None)
    progress(1.0, desc="Voce addestrata e installata!")
    yield (
        log(f"[COMPLETATO] Nuova voce salvata in models/custom/{safe_name} e aggiunta ai menu!"),
        gr.update(choices=labels, value=new_label),
        gr.update(choices=labels, value=new_label),
    )


# ---------------------------------------------------------------------------
# Text-to-Speech (Edge-TTS + RVC v2)
# ---------------------------------------------------------------------------
def load_tts_voices() -> list[str]:
    priority = [
        "it-IT-ElsaNeural",
        "it-IT-IsabellaNeural",
        "it-IT-DiegoNeural",
        "it-IT-GiuseppeMultilingualNeural",
        "en-US-AvaMultilingualNeural",
        "en-US-AndrewMultilingualNeural",
        "ja-JP-NanamiNeural",
        "ja-JP-KeitaNeural",
    ]
    all_voices = list(priority)
    json_path = os.path.join(RVC_DIR, "rvc", "lib", "tools", "tts_voices.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                sn = item.get("ShortName")
                if sn and sn not in all_voices:
                    all_voices.append(sn)
        except Exception:
            pass
    return all_voices


TTS_VOICES_LIST = load_tts_voices()


def generate_tts_rvc(
    text: str,
    rvc_voice_label: str,
    base_tts_voice: str,
    pitch_mode: str,
    pitch_manual: float,
    tts_rate: int,
    index_rate: float,
    progress=gr.Progress(),
):
    logs = []

    def log(msg: str):
        logs.append(msg)
        return "\n".join(logs)

    clean_text = (text or "").strip()
    if not clean_text:
        yield log("[ERRORE] Inserisci un testo da far pronunciare."), None
        return

    if not rvc_voice_label or rvc_voice_label not in MODEL_REGISTRY:
        yield log("[ERRORE] Seleziona una voce RVC valida."), None
        return

    import edge_tts

    tts_dir = os.path.join(OUTPUT_DIR, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_voice = sanitize_name(rvc_voice_label.split("/")[-1], default="voice")

    raw_mp3 = os.path.join(tts_dir, f"_tmp_tts_{ts}.mp3")
    base_wav = os.path.join(tts_dir, f"_tmp_tts_{ts}.wav")
    rvc_wav = os.path.join(tts_dir, f"_tmp_rvc_{ts}.wav")
    final_mp3 = os.path.join(tts_dir, f"tts_{safe_voice}_{ts}.mp3")

    progress(0.15, desc="Sintesi voce base (Edge-TTS)...")
    yield log(f"[TTS 1/3] Sintesi testo con {base_tts_voice} (velocità={int(tts_rate):+d}%)..."), None

    try:
        rate_str = f"{int(tts_rate):+d}%"
        async def _run_edge():
            comm = edge_tts.Communicate(clean_text, base_tts_voice, rate=rate_str)
            await comm.save(raw_mp3)

        asyncio.run(_run_edge())
        (
            ffmpeg
            .input(raw_mp3)
            .output(base_wav, acodec="pcm_s16le", ar=44100, ac=1)
            .overwrite_output()
            .run(cmd=FFMPEG_EXE, quiet=True)
        )
    except Exception as e:
        yield log(f"[ERRORE TTS] {e}"), None
        return

    progress(0.50, desc="Conversione RVC v2 (rmvpe)...")
    v_info = MODEL_REGISTRY[rvc_voice_label]
    pitch, p_desc = resolve_pitch(pitch_mode, pitch_manual, base_wav, rvc_voice_label)
    yield log(f"[TTS 2/3] Inferenza RVC ({rvc_voice_label}) | Pitch: {p_desc} | Index Rate: {index_rate}"), None

    try:
        msg, _ = run_rvc_inference(
            input_wav=base_wav,
            output_wav=rvc_wav,
            pth_path=v_info["pth"],
            index_path=v_info["index"],
            pitch=pitch,
            index_rate=index_rate,
        )
        yield log(f"[TTS 2/3] {msg}"), None

        progress(0.88, desc="Export MP3 320 kbps...")
        (
            ffmpeg
            .input(rvc_wav)
            .output(final_mp3, audio_bitrate="320k", acodec="libmp3lame", ar=44100)
            .overwrite_output()
            .run(cmd=FFMPEG_EXE, quiet=True)
        )
    except Exception as e:
        yield log(f"[ERRORE RVC] {e}"), None
        return
    finally:
        for p in (raw_mp3, base_wav, rvc_wav):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    progress(1.0, desc="Parlato RVC pronto!")
    yield log(f"[COMPLETATO] Audio salvato: {final_mp3}"), final_mp3


# ---------------------------------------------------------------------------
# Main Cover Generation Pipeline
# ---------------------------------------------------------------------------
def generate_cover(
    song_input: str,
    local_file_path: str | None,
    mode: str,
    voice1_label: str,
    pitch1_mode: str,
    pitch1_manual: float,
    voice2_label: str,
    pitch2_mode: str,
    pitch2_manual: float,
    index_rate: float,
    use_reverb: bool,
    progress=gr.Progress(),
):
    logs = []

    def log(msg: str):
        logs.append(msg)
        return "\n".join(logs)

    has_local_file = bool(local_file_path and os.path.exists(local_file_path))
    song_query = (song_input or "").strip()
    if not has_local_file and not song_query:
        yield log("[ERRORE] Inserisci il titolo di una canzone, un URL YouTube oppure carica un file MP3/MP4 locale."), None
        return

    if not voice1_label or voice1_label not in MODEL_REGISTRY:
        yield log("[ERRORE] Seleziona una Voce Principale valida."), None
        return

    is_duet = mode.startswith("Duetto")
    if is_duet and (not voice2_label or voice2_label not in MODEL_REGISTRY):
        yield log("[ERRORE] Seleziona una Seconda Voce valida per la modalità Duetto."), None
        return

    # -----------------------------------------------------------------------
    # STEP 1: Acquire audio (Local MP3/MP4 via ffmpeg OR YouTube via yt-dlp)
    # -----------------------------------------------------------------------
    if has_local_file:
        progress(0.05, desc="Step 1/5: Conversione file locale in WAV...")
        raw_title = Path(local_file_path).stem
        song_folder_name = sanitize_name(raw_title, default="local_song")
        work_dir = os.path.join(OUTPUT_DIR, song_folder_name)
        os.makedirs(work_dir, exist_ok=True)
        original_wav = os.path.join(work_dir, "original.wav")

        yield log(f"[STEP 1/5] File locale rilevato: {os.path.basename(local_file_path)}"), None
        yield log(f"[STEP 1/5] Cartella di lavoro: {work_dir}"), None
        try:
            (
                ffmpeg
                .input(local_file_path)
                .output(original_wav, acodec="pcm_s16le", ar=44100, ac=2)
                .overwrite_output()
                .run(cmd=FFMPEG_EXE, quiet=True)
            )
        except Exception as e:
            yield log(f"[ERRORE] Conversione file locale fallita: {e}"), None
            return
        if not os.path.exists(original_wav):
            yield log("[ERRORE] original.wav non generato dal file locale."), None
            return
        yield log(f"[STEP 1/5] Conversione WAV completata: original.wav ({os.path.getsize(original_wav)/(1024*1024):.2f} MB)"), None
    else:
        progress(0.05, desc="Step 1/5: Download audio con yt-dlp...")
        is_url = song_query.startswith(("http://", "https://")) or "youtube.com" in song_query or "youtu.be" in song_query
        yt_target = song_query if is_url else f"ytsearch1:{song_query}"

        yield log(f"[STEP 1/5] Risoluzione titolo per: {yt_target}"), None
        try:
            title_proc = subprocess.run(
                [YT_DLP_EXE, "--get-title", "--no-playlist", yt_target],
                capture_output=True,
                text=True,
                check=True,
            )
            raw_title = title_proc.stdout.strip().splitlines()[0] if title_proc.stdout.strip() else song_query
        except Exception:
            raw_title = song_query if not is_url else "youtube_song"

        song_folder_name = sanitize_name(raw_title, default="song")
        work_dir = os.path.join(OUTPUT_DIR, song_folder_name)
        os.makedirs(work_dir, exist_ok=True)
        original_wav = os.path.join(work_dir, "original.wav")

        yield log(f"[STEP 1/5] Cartella di lavoro: {work_dir}"), None
        if os.path.exists(original_wav) and os.path.getsize(original_wav) > 10240:
            yield log(f"[STEP 1/5] Cache trovata: uso original.wav già presente ({os.path.getsize(original_wav)/(1024*1024):.2f} MB)"), None
        else:
            if os.path.exists(original_wav):
                os.remove(original_wav)
            dl_cmd = [
                YT_DLP_EXE,
                "--ffmpeg-location", FFMPEG_DIR if os.path.isdir(FFMPEG_DIR) else os.path.dirname(FFMPEG_EXE),
                "--no-playlist",
                "-x",
                "--audio-format", "wav",
                "-o", os.path.join(work_dir, "original.%(ext)s"),
                yt_target,
            ]
            subprocess.run(dl_cmd, check=True)
            if not os.path.exists(original_wav):
                yield log("[ERRORE] Download WAV fallito: original.wav non trovato."), None
                return
            yield log(f"[STEP 1/5] Download completato: original.wav ({os.path.getsize(original_wav)/(1024*1024):.2f} MB)"), None

    # -----------------------------------------------------------------------
    # STEP 2: Demucs separation (htdemucs, --two-stems=vocals)
    # -----------------------------------------------------------------------
    vocals_wav = os.path.join(work_dir, "vocals.wav")
    inst_wav = os.path.join(work_dir, "instrumental.wav")

    if (
        os.path.exists(vocals_wav) and os.path.getsize(vocals_wav) > 10240
        and os.path.exists(inst_wav) and os.path.getsize(inst_wav) > 10240
    ):
        progress(0.45, desc="Step 2/5: Stem Demucs già in cache!")
        yield log("[STEP 2/5] Cache trovata: salto Demucs e uso vocals.wav + instrumental.wav già estratti!"), None
    else:
        progress(0.25, desc="Step 2/5: Separazione voce e base con Demucs...")
        cpu_jobs = str(os.cpu_count() or 4)
        yield log(f"[STEP 2/5] Avvio Demucs (htdemucs --two-stems=vocals, -j {cpu_jobs})..."), None

        sep_out_dir = os.path.join(work_dir, "separated")
        demucs_cmd = [
            DEMUCS_EXE,
            "--two-stems=vocals",
            "-n", "htdemucs",
            "-j", cpu_jobs,
            "-o", sep_out_dir,
            original_wav,
        ]
        subprocess.run(demucs_cmd, check=True)

        demucs_track_dir = os.path.join(sep_out_dir, "htdemucs", "original")
        src_vocals = os.path.join(demucs_track_dir, "vocals.wav")
        src_novocals = os.path.join(demucs_track_dir, "no_vocals.wav")

        if not os.path.exists(src_vocals) or not os.path.exists(src_novocals):
            yield log("[ERRORE] Output Demucs non trovato."), None
            return

        shutil.copy2(src_vocals, vocals_wav)
        shutil.copy2(src_novocals, inst_wav)
        yield log("[STEP 2/5] Separazione completata: vocals.wav + instrumental.wav"), None

    # -----------------------------------------------------------------------
    # STEP 3 & 4: Optional Duet F0 Split + Pure RVC Inference
    # -----------------------------------------------------------------------
    v1_info = MODEL_REGISTRY[voice1_label]

    if not is_duet:
        progress(0.50, desc="Step 3/5: Modalità Voce Singola (nessuno split)...")
        yield log("[STEP 3/5] Modalità Voce Singola: utilizzo diretto di vocals.wav (zero Chorus/vocoder pre-RVC)."), None

        progress(0.60, desc="Step 4/5: Inferenza RVC pura (Voce Principale)...")
        p1, p1_desc = resolve_pitch(pitch1_mode, pitch1_manual, vocals_wav, voice1_label)
        yield log(f"[STEP 4/5] RVC Voce Principale ({voice1_label}) | Pitch: {p1_desc} | Index Rate: {index_rate}"), None

        rvc_out_1 = os.path.join(work_dir, "rvc_voice1.wav")
        msg1, _ = run_rvc_active_only(
            input_wav=vocals_wav,
            output_wav=rvc_out_1,
            pth_path=v1_info["pth"],
            index_path=v1_info["index"],
            pitch=p1,
            index_rate=index_rate,
        )
        yield log(f"[STEP 4/5] RVC completato: {msg1}"), None
        female_intervals = []
    else:
        progress(0.45, desc="Step 3/5: Split F0 Maschile/Femminile...")
        yield log("[STEP 3/5] Analisi F0 + split Maschile/Femminile con crossfade 100ms..."), None

        male_wav = os.path.join(work_dir, "vocals_male.wav")
        female_wav = os.path.join(work_dir, "vocals_female.wav")
        female_intervals, male_med, female_med = split_vocals_by_f0(
            vocals_wav, male_wav, female_wav, fade_ms=100
        )
        yield log(
            f"[STEP 3/5] Split completato: {len(female_intervals)} segmenti femminili "
            f"(F0 globale={male_med:.1f} Hz, F0 femminile={female_med:.1f} Hz)"
        ), None

        progress(0.60, desc="Step 4/5: Inferenza RVC Voce Principale (traccia maschile)...")
        p1, p1_desc = resolve_pitch(pitch1_mode, pitch1_manual, male_wav, voice1_label)
        yield log(f"[STEP 4A/5] RVC Voce Principale ({voice1_label}) | Pitch: {p1_desc}"), None
        rvc_out_1 = os.path.join(work_dir, "rvc_voice1.wav")
        msg1, _ = run_rvc_active_only(
            input_wav=male_wav,
            output_wav=rvc_out_1,
            pth_path=v1_info["pth"],
            index_path=v1_info["index"],
            pitch=p1,
            index_rate=index_rate,
        )
        yield log(f"[STEP 4A/5] Completato: {msg1}"), None

        v2_info = MODEL_REGISTRY[voice2_label]
        rvc_out_2 = os.path.join(work_dir, "rvc_voice2.wav")
        if len(female_intervals) == 0:
            yield log("[STEP 4B/5] Nessun segmento femminile rilevato: traccia 2 silenziosa."), None
            AudioSegment.from_wav(female_wav).export(rvc_out_2, format="wav")
        else:
            progress(0.75, desc="Step 4/5: Inferenza RVC Seconda Voce (traccia femminile)...")
            p2, p2_desc = resolve_pitch(pitch2_mode, pitch2_manual, female_wav, voice2_label)
            yield log(f"[STEP 4B/5] RVC Seconda Voce ({voice2_label}) | Pitch: {p2_desc}"), None
            msg2, _ = run_rvc_active_only(
                input_wav=female_wav,
                output_wav=rvc_out_2,
                pth_path=v2_info["pth"],
                index_path=v2_info["index"],
                pitch=p2,
                index_rate=index_rate,
            )
            yield log(f"[STEP 4B/5] Completato: {msg2}"), None

    # -----------------------------------------------------------------------
    # STEP 5: Mixing with Pedalboard & ffmpeg-python (locked at 0.0s, MP3 320k)
    # -----------------------------------------------------------------------
    progress(0.88, desc="Step 5/5: Mixaggio Pedalboard + Export MP3 320 kbps...")
    yield log(f"[STEP 5/5] Mixaggio a 0.0s | Riverbero Plate={'ON' if use_reverb else 'OFF'} | Nessun AutoTune"), None

    target_sr = 44100
    inst = load_stereo(inst_wav, target_sr)
    orig_vocals = load_stereo(vocals_wav, target_sr)
    v1 = load_stereo(rvc_out_1, target_sr)

    def build_vocal_board(reverb_on: bool) -> Pedalboard:
        chain = [
            HighpassFilter(cutoff_frequency_hz=110.0),
            Compressor(threshold_db=-18.0, ratio=2.8, attack_ms=12.0, release_ms=130.0),
        ]
        if reverb_on:
            chain.append(
                Reverb(room_size=0.60, damping=0.56, wet_level=0.22, dry_level=0.89, width=1.0)
            )
        return Pedalboard(chain)

    orig_act = orig_vocals[:, np.mean(np.abs(orig_vocals), axis=0) > 0.005]
    orig_rms = (
        np.sqrt(np.mean(orig_act ** 2)) if orig_act.size > 0 else np.sqrt(np.mean(orig_vocals ** 2))
    ) + 1e-8

    if not is_duet:
        min_len = min(inst.shape[1], v1.shape[1], orig_vocals.shape[1])
        inst = inst[:, :min_len]
        v1 = v1[:, :min_len]

        v1_fx = build_vocal_board(use_reverb)(v1, target_sr)
        v1_act = v1_fx[:, np.mean(np.abs(v1_fx), axis=0) > 0.005]
        v1_rms = (np.sqrt(np.mean(v1_act ** 2)) if v1_act.size > 0 else np.sqrt(np.mean(v1_fx ** 2))) + 1e-8
        v1_bal = v1_fx * (orig_rms * 0.95 / v1_rms)
        mix = inst * 0.90 + v1_bal
    else:
        v2 = load_stereo(rvc_out_2, target_sr)
        min_len = min(inst.shape[1], v1.shape[1], v2.shape[1], orig_vocals.shape[1])
        inst = inst[:, :min_len]
        v1 = v1[:, :min_len]
        v2 = v2[:, :min_len]

        female_mask = np.zeros(min_len, dtype=np.float32)
        for s_ms, e_ms in female_intervals:
            s_idx = max(0, int(s_ms * target_sr / 1000.0))
            e_idx = min(min_len, int(e_ms * target_sr / 1000.0))
            female_mask[s_idx:e_idx] = 1.0
        male_mask = 1.0 - female_mask

        win = int(0.05 * target_sr)
        if win > 1:
            kernel = np.ones(win, dtype=np.float32) / win
            female_mask = np.convolve(female_mask, kernel, mode="same")
            male_mask = np.convolve(male_mask, kernel, mode="same")

        v1 = v1 * male_mask
        v2 = v2 * female_mask

        v1_fx = build_vocal_board(use_reverb)(v1, target_sr)
        v2_fx = build_vocal_board(use_reverb)(v2, target_sr)

        v1_act = v1_fx[:, np.mean(np.abs(v1_fx), axis=0) > 0.005]
        v1_rms = (np.sqrt(np.mean(v1_act ** 2)) if v1_act.size > 0 else np.sqrt(np.mean(v1_fx ** 2))) + 1e-8
        v1_bal = v1_fx * (orig_rms * 0.95 / v1_rms)

        v2_act = v2_fx[:, np.mean(np.abs(v2_fx), axis=0) > 0.005]
        if v2_act.size > 0:
            v2_rms = np.sqrt(np.mean(v2_act ** 2)) + 1e-8
            v2_bal = v2_fx * (orig_rms * 0.95 * 1.15 / v2_rms)
        else:
            v2_bal = v2_fx

        mix = inst * 0.90 + v1_bal + v2_bal

    master_board = Pedalboard([
        Compressor(threshold_db=-12.0, ratio=1.8, attack_ms=20.0, release_ms=150.0),
        Limiter(threshold_db=-0.8, release_ms=80.0),
    ])
    master_mix = master_board(mix.astype(np.float32), target_sr)

    master_wav = os.path.join(work_dir, f"{song_folder_name}_master.wav")
    final_mp3 = os.path.join(work_dir, f"{song_folder_name}_cover.mp3")
    sf.write(master_wav, master_mix.T, target_sr)

    (
        ffmpeg
        .input(master_wav)
        .output(final_mp3, audio_bitrate="320k", acodec="libmp3lame", ar=target_sr)
        .overwrite_output()
        .run(cmd=FFMPEG_EXE, quiet=True)
    )

    progress(1.0, desc="Cover completata!")
    yield log(f"[COMPLETATO] Cover esportata in MP3 320 kbps: {final_mp3}"), final_mp3


# ---------------------------------------------------------------------------
# Gradio UI - 2000s Kawaii Vocaloid Blog Edition (Zero Emoji)
# ---------------------------------------------------------------------------
MODEL_REGISTRY, INITIAL_LABELS = scan_local_models()
DEFAULT_V1 = INITIAL_LABELS[0] if INITIAL_LABELS else None
DEFAULT_V2 = INITIAL_LABELS[1] if len(INITIAL_LABELS) > 1 else DEFAULT_V1

def _load_data_uri(filename: str, mime: str = "image/gif") -> str:
    path = os.path.join(APP_DIR, "assets", filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


BG_SKY_URI = _load_data_uri("bg_sky.png", "image/png")
GUMI_SPRITE_URI = _load_data_uri("sprite_gumi.png", "image/png")

CUSTOM_CSS = f"""
@keyframes sprite-hop {{
    0%, 100% {{ transform: translateY(0px) rotate(-1.5deg); }}
    50%      {{ transform: translateY(-5px) rotate(1.5deg); }}
}}

@keyframes sprite-roam-lr {{
    0%   {{ left: -50px; transform: scaleX(1); }}
    48%  {{ left: calc(100vw - 5px); transform: scaleX(1); }}
    50%  {{ left: calc(100vw - 5px); transform: scaleX(-1); }}
    98%  {{ left: -50px; transform: scaleX(-1); }}
    100% {{ left: -50px; transform: scaleX(1); }}
}}

@keyframes sprite-roam-rl {{
    0%   {{ right: -50px; transform: scaleX(-1); }}
    48%  {{ right: calc(100vw - 5px); transform: scaleX(-1); }}
    50%  {{ right: calc(100vw - 5px); transform: scaleX(1); }}
    98%  {{ right: -50px; transform: scaleX(1); }}
    100% {{ right: -50px; transform: scaleX(-1); }}
}}

:root, .dark {{
    color-scheme: light !important;
    --body-background-fill: transparent !important;
    --background-fill-primary: #fffdfd !important;
    --background-fill-secondary: #fff7fa !important;
    --block-background-fill: #fffdfd !important;
    --block-label-background-fill: #fff4f8 !important;
    --block-label-text-color: #8c3b68 !important;
    --block-title-text-color: #8c3b68 !important;
    --body-text-color: #6b3b5b !important;
    --body-text-color-subdued: #8f5c7a !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #f2b5cc !important;
    --border-color-primary: #f2b5cc !important;
    --checkbox-background-color: #ffffff !important;
    --checkbox-border-color: #e07ba4 !important;
    --checkbox-label-background-fill: #fff4f8 !important;
    --checkbox-label-text-color: #6b3b5b !important;
    --slider-color: #f29ec0 !important;
}}

html, body {{
    background-color: transparent !important;
    background-image: url("{BG_SKY_URI}") !important;
    background-size: cover !important;
    background-position: center bottom !important;
    background-attachment: fixed !important;
    image-rendering: pixelated;
    font-family: "Verdana", "Trebuchet MS", sans-serif !important;
    color: #6b3b5b !important;
}}

gradio-app, main, .app {{
    background: transparent !important;
    background-color: transparent !important;
}}

footer {{
    display: none !important;
}}

.gradio-container {{
    max-width: 1020px !important;
    margin: 18px auto 70px auto !important;
    padding: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-family: "Verdana", "Trebuchet MS", sans-serif !important;
    color: #6b3b5b !important;
}}

/* Top Journal Header Card */
.journal-header {{
    background: #fffdfd;
    border: 2px solid #f29ec0;
    box-shadow: 4px 4px 0px rgba(242, 158, 192, 0.55);
    padding: 16px 22px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}}

.journal-title {{
    font-family: "Courier New", Consolas, monospace;
    font-size: 24px;
    font-weight: 900;
    letter-spacing: 2px;
    color: #9e4374;
    margin: 0 0 4px 0;
}}

.journal-subtitle {{
    font-size: 12px;
    color: #8f5c7a;
    margin: 0;
}}

.journal-header-right {{
    display: flex;
    align-items: center;
    gap: 14px;
}}

.journal-nav {{
    display: flex;
    gap: 6px;
}}

.journal-nav-pill {{
    font-size: 11px;
    color: #9e4374;
    background: #fff4f8;
    border: 1px solid #f29ec0;
    padding: 3px 9px;
    font-family: "Courier New", monospace;
}}

/* Left Sidebar Card */
.journal-sidebar {{
    background: #fffdfd !important;
    border: 2px solid #f29ec0 !important;
    box-shadow: 4px 4px 0px rgba(242, 158, 192, 0.55) !important;
    padding: 16px !important;
    border-radius: 0px !important;
}}

.sidebar-heading {{
    font-family: "Courier New", Consolas, monospace;
    font-size: 18px;
    font-weight: 900;
    letter-spacing: 2px;
    color: #9e4374;
    margin: 10px 0 6px 0;
    border-bottom: 1.5px dashed #f5bfd4;
    padding-bottom: 3px;
}}

.sidebar-heading:first-child {{
    margin-top: 0;
}}

.sidebar-text {{
    font-size: 11.5px;
    line-height: 1.55;
    color: #7c4d69;
    margin: 0 0 8px 0;
}}

.sidebar-list {{
    list-style: none;
    padding: 0;
    margin: 0 0 10px 0;
    font-size: 11.5px;
    color: #7c4d69;
}}

.sidebar-list li {{
    margin-bottom: 4px;
}}

.status-box {{
    background: #fff4f8;
    border: 1.5px solid #f5bfd4;
    padding: 10px;
    margin-top: 8px;
    font-size: 11px;
    color: #7c4d69;
}}

.status-label {{
    font-family: "Courier New", monospace;
    font-weight: 900;
    font-size: 11px;
    color: #9e4374;
    margin-top: 6px;
}}

.status-label:first-child {{
    margin-top: 0;
}}

.mini-pl-box {{
    background: #ffffff;
    border: 1.5px solid #f29ec0;
    padding: 5px 6px;
    margin: 4px 0;
}}

.mini-pl-title {{
    font-family: "Courier New", monospace;
    font-size: 10.5px;
    font-weight: bold;
    color: #8c3b68;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.mini-pl-btns {{
    gap: 4px !important;
    margin: 2px 0 4px 0 !important;
}}

.mini-pl-btns button {{
    min-width: 0 !important;
    padding: 2px 4px !important;
    font-size: 11px !important;
}}

/* Main Notebook Card (Ruled Paper) */
.journal-main {{
    background-color: #fffdfd !important;
    background-image: repeating-linear-gradient(
        to bottom,
        transparent 0px,
        transparent 27px,
        #f5e4ec 27px,
        #f5e4ec 28px
    ) !important;
    background-position: 0 14px !important;
    background-size: 100% 28px !important;
    border: 2px solid #f29ec0 !important;
    border-top: 10px solid #fce0ec !important;
    box-shadow: 4px 4px 0px rgba(242, 158, 192, 0.55) !important;
    padding: 14px 18px 18px 18px !important;
    border-radius: 0px !important;
}}

.journal-entry-meta {{
    font-family: "Courier New", monospace;
    font-size: 11px;
    line-height: 28px;
    height: 28px;
    color: #b07a98;
    margin: 0;
    padding: 0;
}}

.journal-entry-title {{
    font-family: "Courier New", monospace;
    font-size: 17px;
    font-weight: 900;
    line-height: 28px;
    height: 28px;
    color: #9e4374;
    margin: 0 0 14px 0;
    padding: 0;
}}

.journal-subbox {{
    background: rgba(255, 250, 252, 0.92) !important;
    border: 1.5px dashed #f2b5cc !important;
    padding: 10px !important;
    border-radius: 0px !important;
}}

/* Soft Pastel Buttons */
button.primary, button.lg {{
    background: linear-gradient(180deg, #ffe6f0 0%, #f79ec1 100%) !important;
    color: #6b2248 !important;
    font-family: "Courier New", monospace !important;
    font-weight: 900 !important;
    letter-spacing: 1px !important;
    border: 1.5px solid #e07ba4 !important;
    box-shadow: 2px 2px 0px #f2b5cc !important;
    border-radius: 2px !important;
}}

button.secondary {{
    background: #fff4f8 !important;
    color: #8c3b68 !important;
    font-family: "Courier New", monospace !important;
    font-weight: bold !important;
    border: 1.5px solid #f29ec0 !important;
    box-shadow: 2px 2px 0px #f5d0df !important;
    border-radius: 2px !important;
}}

button:active {{
    transform: translate(1px, 1px) !important;
    box-shadow: 1px 1px 0px #f2b5cc !important;
}}

/* Tabs */
.tab-nav button {{
    font-family: "Courier New", monospace !important;
    font-size: 12px !important;
    font-weight: bold !important;
    border: 1.5px solid #f29ec0 !important;
    background: #fff4f8 !important;
    color: #9e4374 !important;
    margin-right: 4px !important;
    border-radius: 0px !important;
    padding: 4px 12px !important;
}}

.tab-nav button.selected {{
    background: #f7b2ce !important;
    color: #5c193d !important;
}}

/* Notebook Log Box */
.journal-log textarea {{
    font-family: "Courier New", Consolas, monospace !important;
    font-size: 12px !important;
    background: #fff7fa !important;
    color: #7a345a !important;
    border: 1.5px solid #f2b5cc !important;
}}

/* Roaming Pixel-Art Sprites */
.chibi-stage {{
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9999;
    overflow: hidden;
}}

.vocaloid-walker {{
    position: fixed;
    left: 40px;
    top: 0px;
    width: 44px;
    opacity: 0.96;
    pointer-events: auto;
    cursor: grab;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
    will-change: transform;
}}

.vocaloid-walker.dragging {{
    cursor: grabbing;
}}

.chibi-pixel-sprite {{
    width: 44px;
    height: auto;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
    display: block;
    pointer-events: none;
    animation: sprite-hop 0.55s ease-in-out infinite;
}}

.vocaloid-walker.dragging .chibi-pixel-sprite {{
    animation: none;
}}

.vocaloid-walker.happy-spin .chibi-pixel-sprite {{
    transition: transform 0.38s ease;
    transform: rotate(360deg) scale(1.15);
}}
"""

HEADER_HTML = """
<div class="journal-header">
    <div>
        <div class="journal-title">ai cover studio</div>
        <p class="journal-subtitle">little vocaloid covers, rvc v2 models, and cozy mixes.</p>
    </div>
    <div class="journal-header-right">
        <svg width="54" height="44" viewBox="0 0 54 44" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <path d="M26 19 C16 6, 4 8, 5 20 C6 30, 18 26, 26 21 Z" fill="#ffd1e3" stroke="#e88ab0" stroke-width="2"/>
            <path d="M28 19 C38 6, 50 8, 49 20 C48 30, 36 26, 28 21 Z" fill="#ffd1e3" stroke="#e88ab0" stroke-width="2"/>
            <path d="M24 23 L14 41 L21 38 L26 24 Z" fill="#ffb8d4" stroke="#e88ab0" stroke-width="1.8"/>
            <path d="M30 23 L40 41 L33 38 L28 24 Z" fill="#ffb8d4" stroke="#e88ab0" stroke-width="1.8"/>
            <ellipse cx="27" cy="21" rx="5" ry="4.5" fill="#ffb8d4" stroke="#e88ab0" stroke-width="1.8"/>
        </svg>
        <div class="journal-nav">
            <span class="journal-nav-pill">studio</span>
            <span class="journal-nav-pill">models</span>
            <span class="journal-nav-pill">320kbps</span>
        </div>
    </div>
</div>
"""

SIDEBAR_HTML = f"""
<div class="sidebar-heading">ABOUT</div>
<p class="sidebar-text">
    ciao! sono <b>Meko</b> e questo è il mio studio personale di <b>AI Cover</b>.
    Nasce per creare cover Vocaloid e personalizzate in locale da YouTube o da file MP3/MP4 sul PC, senza passare da servizi online.
</p>

<div class="sidebar-heading">HOW IT WORKS</div>
<ul class="sidebar-list">
    <li>1. incolla un titolo/url youtube o carica un file locale (mp3, mp4, wav).</li>
    <li>2. <code>demucs</code> separa voce e base strumentale.</li>
    <li>3. in duetto, <code>pyworld</code> divide voce maschile e femminile via F0.</li>
    <li>4. <code>rvc v2 (rmvpe)</code> converte il timbro (scarica nuove voci da HF, Voice-Models, GitHub o URL diretto).</li>
    <li>5. <code>pedalboard</code> + <code>ffmpeg</code> mixano in mp3 320 kbps, salvati nella tab Archivio.</li>
</ul>

<div class="sidebar-heading">MADE WITH</div>
<ul class="sidebar-list">
    <li>* python + gradio</li>
    <li>* rvc_cli + htdemucs</li>
    <li>* pyworld + pydub</li>
    <li>* pedalboard + ffmpeg</li>
    <li>* hf + voice-models</li>
</ul>

<div class="sidebar-heading">STATUS</div>
<div class="status-box">
    <div class="status-label">WEBMASTER</div>
    <div>meko</div>
    <div class="status-label">LOCAL MODELS</div>
    <div>{len(INITIAL_LABELS)} voices ready</div>
    <div class="status-label">ENGINE</div>
    <div>cpu optimized (rmvpe)</div>
</div>
"""

CHIBI_STAGE_HTML = f"""
<div class="chibi-stage" aria-hidden="true">
    <div id="gumi-pet" class="vocaloid-walker" title="GUMI">
        <img class="chibi-pixel-sprite" src="{GUMI_SPRITE_URI}" alt="GUMI" draggable="false" />
    </div>
</div>
"""

with gr.Blocks(title="ai cover studio // kawaii journal") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=240, elem_classes=["journal-sidebar"]):
            gr.HTML(SIDEBAR_HTML)

        with gr.Column(scale=3, elem_classes=["journal-main"]):
            gr.HTML(
                '<div class="journal-entry-meta">entry #01 // rvc v2 local studio</div>'
                '<div class="journal-entry-title">cover generator &amp; voice archive</div>'
            )

            with gr.Tabs():
                # TAB 1: Generatore Cover
                with gr.Tab("Generatore Cover"):
                    song_input = gr.Textbox(
                        label="Titolo canzone o URL YouTube",
                        placeholder="Es: Niko Pandetta - Pistole nella Fendi  oppure  https://www.youtube.com/watch?v=...",
                    )
                    local_file_input = gr.File(
                        label="Oppure carica un file locale dal PC (MP3 / MP4 / WAV)",
                        file_types=[".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg"],
                        type="filepath",
                    )

                    mode_radio = gr.Radio(
                        choices=["Voce Singola", "Duetto (Split Maschile / Femminile via F0)"],
                        value="Voce Singola",
                        label="Modalita",
                    )

                    with gr.Row():
                        with gr.Column(elem_classes=["journal-subbox"]):
                            with gr.Row():
                                voice1_dd = gr.Dropdown(
                                    choices=INITIAL_LABELS,
                                    value=DEFAULT_V1,
                                    label="Voce Principale",
                                    scale=4,
                                )
                                refresh_btn = gr.Button("Aggiorna lista", scale=1, variant="secondary")
                            pitch1_mode = gr.Radio(
                                choices=["Auto", "-12", "0", "+12", "Manuale"],
                                value="Auto",
                                label="Pitch Voce Principale",
                            )
                            pitch1_manual = gr.Slider(
                                minimum=-24,
                                maximum=24,
                                value=0,
                                step=1,
                                label="Pitch Manuale (semitoni)",
                                visible=False,
                            )

                        with gr.Column(visible=False, elem_classes=["journal-subbox"]) as duet_col:
                            voice2_dd = gr.Dropdown(
                                choices=INITIAL_LABELS,
                                value=DEFAULT_V2,
                                label="Seconda Voce (per Duetto)",
                            )
                            pitch2_mode = gr.Radio(
                                choices=["Auto", "-12", "0", "+12", "Manuale"],
                                value="Auto",
                                label="Pitch Seconda Voce",
                            )
                            pitch2_manual = gr.Slider(
                                minimum=-24,
                                maximum=24,
                                value=0,
                                step=1,
                                label="Pitch Manuale 2a Voce (semitoni)",
                                visible=False,
                            )

                    with gr.Row(elem_classes=["journal-subbox"]):
                        index_rate_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.75,
                            step=0.05,
                            label="Index Rate",
                        )
                        reverb_cb = gr.Checkbox(
                            value=True,
                            label="Riverbero Plate",
                        )

                    generate_btn = gr.Button("Genera Cover", variant="primary", size="lg")

                    log_box = gr.Textbox(
                        label="Log di Avanzamento",
                        lines=11,
                        interactive=False,
                        elem_classes=["journal-log"],
                    )
                    audio_out = gr.Audio(
                        label="Cover Finale (MP3 320 kbps)",
                        type="filepath",
                    )

                # TAB 2: Scarica Nuova Voce
                with gr.Tab("Scarica Nuova Voce"):
                    hf_candidates_state = gr.State({})
                    with gr.Column(elem_classes=["journal-subbox"]):
                        gr.HTML(
                            '<p style="font-size:12px;margin:0 0 8px 0;"><b>Step 1:</b> Cerca su <b>Hugging Face (Models &amp; Datasets)</b>, '
                            '<b>Voice-Models.com</b> e <b>GitHub</b>, poi seleziona il modello esatto prima di confermare il download.</p>'
                        )
                        with gr.Row():
                            hf_name_input = gr.Textbox(
                                label="Nome Personaggio / Artista",
                                placeholder="Es: Kasane Teto, Sfera Ebbasta, Hatsune Miku...",
                                scale=4,
                            )
                            hf_search_btn = gr.Button("Cerca Modelli", variant="secondary", scale=1)

                        hf_results_dd = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Modelli trovati (seleziona prima di confermare)",
                            visible=False,
                        )
                        hf_confirm_btn = gr.Button(
                            "Conferma e Scarica Selezionato",
                            variant="primary",
                            visible=False,
                        )

                        gr.HTML(
                            '<hr style="border:none;border-top:1px dashed #f2b5cc;margin:10px 0;">'
                            '<p style="font-size:12px;margin:0 0 6px 0;"><b>Oppure installa da URL diretto</b> '
                            '(.zip o .pth da qualsiasi sito come weights.gg, pixeldrain, github, ecc.):</p>'
                        )
                        with gr.Row():
                            direct_url_input = gr.Textbox(
                                label="URL diretto (.zip o .pth)",
                                placeholder="https://.../modello_rvc_v2.zip",
                                scale=3,
                            )
                            direct_name_input = gr.Textbox(
                                label="Nome cartella modello",
                                placeholder="Es: mio_modello",
                                scale=1,
                            )
                        direct_download_btn = gr.Button("Scarica da URL Diretto", variant="secondary")

                        hf_log_box = gr.Textbox(
                            label="Stato Ricerca e Download",
                            lines=9,
                            interactive=False,
                            elem_classes=["journal-log"],
                        )

                # TAB 3: Crea Voce (Training RVC v2)
                with gr.Tab("Crea Voce (Training)"):
                    with gr.Column(elem_classes=["journal-subbox"]):
                        gr.HTML(
                            '<p style="font-size:12px;margin:0 0 8px 0;">Addestra un nuovo modello vocale <b>RVC v2</b> dai tuoi file audio. '
                            '<i>Nota: su PC senza GPU NVIDIA il training gira su CPU ed e lento (consigliate 20-50 epoche su 2-5 min di voce pulita per i test).</i></p>'
                        )
                        train_name_input = gr.Textbox(
                            label="Nome della nuova voce",
                            placeholder="Es: mia_voce_custom",
                        )
                        train_files_input = gr.File(
                            label="File audio/video per il dataset (WAV / MP3 / MP4)",
                            file_count="multiple",
                            file_types=[".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg"],
                            type="filepath",
                        )
                        train_demucs_cb = gr.Checkbox(
                            value=False,
                            label="Estrai prima la voce con Demucs (attiva se i file contengono musica di sottofondo)",
                        )
                        with gr.Row():
                            train_sr_radio = gr.Radio(
                                choices=["40000", "48000"],
                                value="40000",
                                label="Sample Rate",
                            )
                            train_epochs_slider = gr.Slider(
                                minimum=10,
                                maximum=500,
                                value=50,
                                step=10,
                                label="Epoche Totali",
                            )
                        with gr.Row():
                            train_batch_slider = gr.Slider(
                                minimum=1,
                                maximum=16,
                                value=4,
                                step=1,
                                label="Batch Size",
                            )
                            train_save_slider = gr.Slider(
                                minimum=5,
                                maximum=100,
                                value=10,
                                step=5,
                                label="Salva checkpoint ogni N epoche",
                            )
                        train_btn = gr.Button("Avvia Training RVC v2", variant="primary")
                        train_log_box = gr.Textbox(
                            label="Log Training RVC v2",
                            lines=10,
                            interactive=False,
                            elem_classes=["journal-log"],
                        )

                # TAB 4: Text to Speech (TTS + RVC)
                with gr.Tab("Text to Speech") as tts_tab:
                    with gr.Column(elem_classes=["journal-subbox"]):
                        gr.HTML(
                            '<p style="font-size:12px;margin:0 0 8px 0;">Scrivi un testo e fallo pronunciare da una delle tue voci <b>RVC v2</b>.</p>'
                        )
                        tts_text_input = gr.Textbox(
                            label="Testo da far dire alla voce RVC",
                            placeholder="Es: Ciao! Sono Miku e questo e AI Cover Studio!",
                            lines=4,
                        )
                        with gr.Row():
                            tts_rvc_dd = gr.Dropdown(
                                choices=INITIAL_LABELS,
                                value=DEFAULT_V1,
                                label="Voce RVC di destinazione",
                            )
                            tts_base_dd = gr.Dropdown(
                                choices=TTS_VOICES_LIST,
                                value="it-IT-ElsaNeural",
                                label="Voce Base TTS (lingua e pronuncia)",
                            )
                        with gr.Row():
                            tts_pitch_mode = gr.Radio(
                                choices=["Auto", "-12", "0", "+12", "Manuale"],
                                value="Auto",
                                label="Pitch",
                            )
                            tts_pitch_manual = gr.Slider(
                                minimum=-24,
                                maximum=24,
                                value=0,
                                step=1,
                                label="Pitch Manuale (semitoni)",
                                visible=False,
                            )
                        with gr.Row():
                            tts_rate_slider = gr.Slider(
                                minimum=-50,
                                maximum=50,
                                value=0,
                                step=5,
                                label="Velocita parlato (%)",
                            )
                            tts_index_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.75,
                                step=0.05,
                                label="Index Rate",
                            )
                        tts_btn = gr.Button("Genera Parlato RVC", variant="primary")
                        tts_log_box = gr.Textbox(
                            label="Log Text to Speech",
                            lines=6,
                            interactive=False,
                            elem_classes=["journal-log"],
                        )
                        tts_audio_out = gr.Audio(
                            label="Audio TTS Finale (MP3 320 kbps)",
                            type="filepath",
                        )

                # TAB 5: Archivio Cover Generate
                with gr.Tab("Archivio") as archive_tab:
                    with gr.Column(elem_classes=["journal-subbox"]):
                        gr.HTML(
                            '<p style="font-size:12px;margin:0 0 8px 0;">Tutti i file MP3 generati in precedenza. '
                            'Seleziona una traccia per ascoltarla o scaricarla.</p>'
                        )
                        ARCHIVE_REGISTRY, ARCHIVE_LABELS = scan_archive_files()
                        DEFAULT_ARCH = ARCHIVE_LABELS[0] if ARCHIVE_LABELS else None
                        DEFAULT_ARCH_PATH = ARCHIVE_REGISTRY.get(DEFAULT_ARCH) if DEFAULT_ARCH else None
                        with gr.Row():
                            archive_dd = gr.Dropdown(
                                choices=ARCHIVE_LABELS,
                                value=DEFAULT_ARCH,
                                label="Cover Salvate",
                                scale=4,
                            )
                            archive_refresh_btn = gr.Button("Aggiorna Archivio", scale=1, variant="secondary")
                        archive_audio = gr.Audio(
                            label="Ascolta dall'Archivio",
                            value=DEFAULT_ARCH_PATH,
                            type="filepath",
                        )
                        archive_file = gr.File(
                            label="Scarica File MP3",
                            value=DEFAULT_ARCH_PATH,
                        )

    gr.HTML(CHIBI_STAGE_HTML)

    # Dynamic UI Visibility
    mode_radio.change(
        fn=lambda m: gr.update(visible=m.startswith("Duetto")),
        inputs=[mode_radio],
        outputs=[duet_col],
    )
    pitch1_mode.change(
        fn=lambda p: gr.update(visible=(p == "Manuale")),
        inputs=[pitch1_mode],
        outputs=[pitch1_manual],
    )
    pitch2_mode.change(
        fn=lambda p: gr.update(visible=(p == "Manuale")),
        inputs=[pitch2_mode],
        outputs=[pitch2_manual],
    )
    tts_pitch_mode.change(
        fn=lambda p: gr.update(visible=(p == "Manuale")),
        inputs=[tts_pitch_mode],
        outputs=[tts_pitch_manual],
    )

    # Actions
    refresh_btn.click(
        fn=refresh_models_ui,
        inputs=[],
        outputs=[voice1_dd, voice2_dd],
    )

    hf_search_btn.click(
        fn=search_rvc_models,
        inputs=[hf_name_input],
        outputs=[hf_log_box, hf_candidates_state, hf_results_dd, hf_confirm_btn],
    )

    hf_confirm_btn.click(
        fn=confirm_download_model,
        inputs=[hf_results_dd, hf_candidates_state, hf_name_input],
        outputs=[hf_log_box, voice1_dd, voice2_dd],
    )

    direct_download_btn.click(
        fn=install_from_direct_url,
        inputs=[direct_url_input, direct_name_input],
        outputs=[hf_log_box, voice1_dd, voice2_dd],
    )

    train_btn.click(
        fn=train_rvc_voice,
        inputs=[
            train_name_input,
            train_files_input,
            train_demucs_cb,
            train_sr_radio,
            train_epochs_slider,
            train_batch_slider,
            train_save_slider,
        ],
        outputs=[train_log_box, voice1_dd, voice2_dd],
    )

    tts_btn.click(
        fn=generate_tts_rvc,
        inputs=[
            tts_text_input,
            tts_rvc_dd,
            tts_base_dd,
            tts_pitch_mode,
            tts_pitch_manual,
            tts_rate_slider,
            tts_index_slider,
        ],
        outputs=[tts_log_box, tts_audio_out],
    )

    tts_tab.select(
        fn=lambda: gr.update(choices=scan_local_models()[1]),
        inputs=[],
        outputs=[tts_rvc_dd],
    )

    archive_refresh_btn.click(
        fn=refresh_archive_ui,
        inputs=[],
        outputs=[archive_dd, archive_audio, archive_file],
    )

    archive_dd.change(
        fn=select_archive_file,
        inputs=[archive_dd],
        outputs=[archive_audio, archive_file],
    )

    archive_tab.select(
        fn=refresh_archive_ui,
        inputs=[],
        outputs=[archive_dd, archive_audio, archive_file],
    )

    generate_btn.click(
        fn=generate_cover,
        inputs=[
            song_input,
            local_file_input,
            mode_radio,
            voice1_dd,
            pitch1_mode,
            pitch1_manual,
            voice2_dd,
            pitch2_mode,
            pitch2_manual,
            index_rate_slider,
            reverb_cb,
        ],
        outputs=[log_box, audio_out],
    )


FORCE_LIGHT_HEAD = """
<script>
(function() {
    try { localStorage.setItem('theme', 'light'); } catch (e) {}
    function stripDark() {
        document.documentElement.classList.remove('dark');
        if (document.body) document.body.classList.remove('dark');
    }
    stripDark();
    new MutationObserver(stripDark).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    window.addEventListener('DOMContentLoaded', stripDark);

    // Interactive Shimeji-style GUMI pet
    function initGumiPet() {
        const pet = document.getElementById('gumi-pet');
        if (!pet || pet.dataset.bound === '1') return;
        pet.dataset.bound = '1';

        const W = 44;
        const H = 50;
        let x = 60;
        let y = Math.max(0, window.innerHeight - H - 6);
        let vx = 1.15;
        let vy = 0;
        let dir = 1;
        let dragging = false;
        let movedDuringDrag = false;
        let dragOffsetX = 0;
        let dragOffsetY = 0;
        let lastPointerX = 0;
        let lastPointerY = 0;
        let tilt = 0;
        let audioCtx = null;
        let lastSoundTime = 0;

        function playGumiImpactSound(intensity) {
            const nowMs = performance.now();
            if (nowMs - lastSoundTime < 350) return;
            lastSoundTime = nowMs;
            try {
                if (!audioCtx) {
                    const Ctx = window.AudioContext || window.webkitAudioContext;
                    if (!Ctx) return;
                    audioCtx = new Ctx();
                }
                if (audioCtx.state === 'suspended') audioCtx.resume();

                const t = audioCtx.currentTime;
                // Formant-filtered cute Vocaloid squeak + comedic censor bleep
                const osc = audioCtx.createOscillator();
                const filter = audioCtx.createBiquadFilter();
                const gain = audioCtx.createGain();

                osc.type = 'sawtooth';
                const baseF = 520 + Math.min(260, intensity * 18);
                osc.frequency.setValueAtTime(baseF, t);
                osc.frequency.exponentialRampToValueAtTime(baseF * 1.35, t + 0.07);
                osc.frequency.exponentialRampToValueAtTime(baseF * 0.72, t + 0.16);

                filter.type = 'bandpass';
                filter.frequency.setValueAtTime(1450, t);
                filter.Q.setValueAtTime(4.5, t);

                gain.gain.setValueAtTime(0.001, t);
                gain.gain.linearRampToValueAtTime(0.16, t + 0.02);
                gain.gain.exponentialRampToValueAtTime(0.001, t + 0.18);

                osc.connect(filter);
                filter.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start(t);
                osc.stop(t + 0.19);

                // Comedic censor bleep right after impact
                const bleep = audioCtx.createOscillator();
                const bGain = audioCtx.createGain();
                bleep.type = 'sine';
                bleep.frequency.setValueAtTime(1000, t + 0.14);
                bGain.gain.setValueAtTime(0.001, t + 0.14);
                bGain.gain.linearRampToValueAtTime(0.11, t + 0.16);
                bGain.gain.setValueAtTime(0.11, t + 0.34);
                bGain.gain.exponentialRampToValueAtTime(0.001, t + 0.38);
                bleep.connect(bGain);
                bGain.connect(audioCtx.destination);
                bleep.start(t + 0.14);
                bleep.stop(t + 0.39);
            } catch (err) {}
        }

        function floorY() {
            return Math.max(0, window.innerHeight - H - 6);
        }

        pet.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            dragging = true;
            movedDuringDrag = false;
            pet.classList.add('dragging');
            pet.setPointerCapture(e.pointerId);
            dragOffsetX = e.clientX - x;
            dragOffsetY = e.clientY - y;
            lastPointerX = e.clientX;
            lastPointerY = e.clientY;
            vx = 0;
            vy = 0;
        });

        pet.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - lastPointerX;
            const dy = e.clientY - lastPointerY;
            if (Math.abs(dx) > 2 || Math.abs(dy) > 2) movedDuringDrag = true;
            x = Math.min(window.innerWidth - W, Math.max(0, e.clientX - dragOffsetX));
            y = Math.min(floorY(), Math.max(0, e.clientY - dragOffsetY));
            vx = dx * 0.45;
            vy = dy * 0.45;
            if (Math.abs(dx) > 1) dir = dx >= 0 ? 1 : -1;
            tilt = Math.max(-22, Math.min(22, dx * 1.4));
            lastPointerX = e.clientX;
            lastPointerY = e.clientY;
        });

        function releasePet(e) {
            if (!dragging) return;
            dragging = false;
            pet.classList.remove('dragging');
            tilt = 0;
            if (!movedDuringDrag) {
                // Click / pet: happy jump & spin!
                vy = -8.5;
                vx = dir * 1.5;
                pet.classList.remove('happy-spin');
                void pet.offsetWidth;
                pet.classList.add('happy-spin');
                setTimeout(() => pet.classList.remove('happy-spin'), 420);
            }
        }

        pet.addEventListener('pointerup', releasePet);
        pet.addEventListener('pointercancel', releasePet);

        pet.addEventListener('dblclick', (e) => {
            e.preventDefault();
            vy = -11;
            dir = -dir;
            vx = dir * 2.2;
        });

        function tick() {
            const fy = floorY();
            if (!dragging) {
                if (y < fy || vy < 0) {
                    vy += 0.55; // gravity
                    x += vx;
                    y += vy;
                    vx *= 0.985;
                    if (y >= fy) {
                        y = fy;
                        if (Math.abs(vy) > 3.2) {
                            playGumiImpactSound(Math.abs(vy));
                            vy = -vy * 0.35; // soft bounce
                        } else {
                            vy = 0;
                            vx = dir * 1.15;
                        }
                    }
                } else {
                    y = fy;
                    x += vx;
                    if (Math.abs(vx) < 0.8) vx = dir * 1.15;
                }

                if (x <= 0) {
                    if (Math.abs(vx) > 2.8) playGumiImpactSound(Math.abs(vx));
                    x = 0;
                    dir = 1;
                    vx = Math.abs(vx) || 1.15;
                } else if (x >= window.innerWidth - W) {
                    if (Math.abs(vx) > 2.8) playGumiImpactSound(Math.abs(vx));
                    x = window.innerWidth - W;
                    dir = -1;
                    vx = -(Math.abs(vx) || 1.15);
                }
            }

            pet.style.left = '0px';
            pet.style.top = '0px';
            pet.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) scaleX(${dir}) rotate(${tilt.toFixed(1)}deg)`;
            requestAnimationFrame(tick);
        }

        requestAnimationFrame(tick);
    }

    const observer = new MutationObserver(initGumiPet);
    observer.observe(document.documentElement, { childList: true, subtree: true });
    window.addEventListener('DOMContentLoaded', initGumiPet);
})();
</script>
"""

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, css=CUSTOM_CSS, head=FORCE_LIGHT_HEAD)

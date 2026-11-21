import os
import sys
import shutil
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from pedalboard import (
    Pedalboard,
    Reverb,
    Compressor,
    HighpassFilter,
    LowShelfFilter,
    PeakFilter,
    HighShelfFilter,
    Limiter,
)

project_root = r"C:\Users\seren\Desktop\Progetto_Pandetta_Miku"
os.chdir(project_root)

ffmpeg_dir = os.path.join(project_root, "venv", "Lib", "site-packages", "static_ffmpeg", "bin", "win32")
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
ffprobe_exe = os.path.join(ffmpeg_dir, "ffprobe.exe")

AudioSegment.converter = ffmpeg_exe
AudioSegment.ffprobe = ffprobe_exe


def mute_intervals_pydub(audio: AudioSegment, mute_ranges_ms: list[tuple[int, int]], fade_ms: int = 250) -> AudioSegment:
    """
    Applies silence (mute) to specific millisecond intervals on a pydub AudioSegment,
    with 250ms fade-out (before mute) and 250ms fade-in (after mute) to avoid digital clicks.
    """
    total_len = len(audio)
    sorted_ranges = sorted(mute_ranges_ms, key=lambda x: x[0])
    merged = []
    for start, end in sorted_ranges:
        start = max(0, min(total_len, start))
        end = max(0, min(total_len, end))
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    result = AudioSegment.empty()
    cursor = 0

    for m_start, m_end in merged:
        if m_start > cursor:
            active_seg = audio[cursor:m_start]
            if cursor > 0 and len(active_seg) > 0:
                f_in = min(fade_ms, len(active_seg))
                active_seg = active_seg.fade_in(f_in)
            if m_start < total_len and len(active_seg) > 0:
                f_out = min(fade_ms, len(active_seg))
                active_seg = active_seg.fade_out(f_out)
            result += active_seg

        silence_dur = m_end - m_start
        if silence_dur > 0:
            result += AudioSegment.silent(
                duration=silence_dur,
                frame_rate=audio.frame_rate,
            ).set_channels(audio.channels).set_sample_width(audio.sample_width)

        cursor = m_end

    if cursor < total_len:
        tail_seg = audio[cursor:total_len]
        if cursor > 0 and len(tail_seg) > 0:
            f_in = min(fade_ms, len(tail_seg))
            tail_seg = tail_seg.fade_in(f_in)
        result += tail_seg

    return result


def load_stereo(path: str, target_sr: int = 44100) -> np.ndarray:
    y, sr = sf.read(path, always_2d=True)
    y = y.T.astype(np.float32)  # (channels, samples)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    elif y.shape[0] > 2:
        y = y[:2, :]
    return y


def main():
    target_sr = 44100
    fade_ms = 250

    print("[FASE 6] Loading vocal tracks with pydub for exact duet script...", flush=True)
    miku_seg = AudioSegment.from_wav("miku_vocals.wav")
    pand_seg = AudioSegment.from_wav("pandetta_vocals.wav")

    # Exact Duet Script (in milliseconds):
    # - PANDETTA CANTA (Miku Muta): 0:00 -> 0:34 (0 - 34000 ms), 1:04 -> 1:21 (64000 - 81000 ms)
    # - MIKU CANTA (Pandetta Muto): 0:34 -> 0:49 (34000 - 49000 ms), 1:21 -> 1:36 (81000 - 96000 ms)
    # - ENTRAMBI CANTANO (Nessun Muto - Ritornelli): 0:49 -> 1:04 (49000 - 64000 ms), 1:36 -> fine (96000 - end)

    miku_mute_ranges = [
        (0, 34000),      # 0:00 -> 0:34
        (64000, 81000),  # 1:04 -> 1:21
    ]

    pandetta_mute_ranges = [
        (34000, 49000),  # 0:34 -> 0:49
        (81000, 96000),  # 1:21 -> 1:36
    ]

    print(f"[FASE 6] Muting Miku intervals (ms): {miku_mute_ranges} with {fade_ms}ms fade-in/out...", flush=True)
    miku_arranged = mute_intervals_pydub(miku_seg, miku_mute_ranges, fade_ms=fade_ms)

    print(f"[FASE 6] Muting Pandetta intervals (ms): {pandetta_mute_ranges} with {fade_ms}ms fade-in/out...", flush=True)
    pand_arranged = mute_intervals_pydub(pand_seg, pandetta_mute_ranges, fade_ms=fade_ms)

    miku_arr_path = "miku_vocals_arranged.wav"
    pand_arr_path = "pandetta_vocals_arranged.wav"
    miku_arranged.export(miku_arr_path, format="wav")
    pand_arranged.export(pand_arr_path, format="wav")
    print(f"[FASE 6] Exported arranged vocal stems: {miku_arr_path}, {pand_arr_path}", flush=True)

    print("[FASE 6] Loading instrumental.wav and arranged vocal stems (aligned at 0.0s)...", flush=True)
    inst = load_stereo("instrumental.wav", target_sr)
    miku = load_stereo(miku_arr_path, target_sr)
    pand = load_stereo(pand_arr_path, target_sr)

    min_len = min(inst.shape[1], miku.shape[1], pand.shape[1])
    inst = inst[:, :min_len]
    miku = miku[:, :min_len]
    pand = pand[:, :min_len]

    miku_full = load_stereo("miku_vocals.wav", target_sr)[:, :min_len]
    pand_full = load_stereo("pandetta_vocals.wav", target_sr)[:, :min_len]

    # Warm Plate Reverb + natural analog-style EQ/Compression on Pandetta (NO AutoTune)
    print("[FASE 6] Applying Warm Plate Reverb + Compression chain to pandetta_vocals (NO AutoTune)...", flush=True)
    pandetta_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=90.0),
        LowShelfFilter(cutoff_frequency_hz=220.0, gain_db=1.5),       # Warm chest body
        PeakFilter(cutoff_frequency_hz=2400.0, gain_db=1.5, q=0.9),  # Natural vocal presence
        HighShelfFilter(cutoff_frequency_hz=7500.0, gain_db=1.0),    # Smooth plate air
        Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=15.0, release_ms=140.0),
        # Warm Plate Reverb configuration (dense, warm damping, wide stereo plate)
        Reverb(room_size=0.62, damping=0.60, wet_level=0.25, dry_level=0.88, width=1.0),
    ])
    pand_fx = pandetta_board(pand, target_sr)

    miku_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=120.0),
        Compressor(threshold_db=-16.0, ratio=2.8, attack_ms=12.0, release_ms=110.0),
        Reverb(room_size=0.48, damping=0.55, wet_level=0.16, dry_level=0.92, width=0.9),
    ])
    miku_fx = miku_board(miku, target_sr)

    # Gentle stereo panning: Miku slightly Left (L), Pandetta slightly Right (R)
    print("[FASE 6] Applying gentle stereo panning: Miku slightly Left (L), Pandetta slightly Right (R)...", flush=True)
    miku_panned = np.vstack([miku_fx[0] * 1.08, miku_fx[1] * 0.82])
    pand_panned = np.vstack([pand_fx[0] * 0.82, pand_fx[1] * 1.08])

    inst_rms = np.sqrt(np.mean(inst ** 2)) + 1e-8
    miku_ref_rms = np.sqrt(np.mean(miku_full ** 2)) + 1e-8
    pand_ref_rms = np.sqrt(np.mean(pand_full ** 2)) + 1e-8

    miku_norm = miku_panned * (inst_rms * 0.76 / miku_ref_rms)
    pand_norm = pand_panned * (inst_rms * 0.68 / pand_ref_rms)

    mix = inst * 0.88 + miku_norm + pand_norm

    master_board = Pedalboard([
        Compressor(threshold_db=-12.0, ratio=1.8, attack_ms=20.0, release_ms=160.0),
        Limiter(threshold_db=-0.8, release_ms=80.0),
    ])
    master_mix = master_board(mix.astype(np.float32), target_sr)

    temp_wav = "Miku_Pandetta_Cantarella_Final_master.wav"
    out_mp3 = "Miku_Pandetta_Cantarella_Final.mp3"
    sf.write(temp_wav, master_mix.T, target_sr)
    print(f"[FASE 6] Saved master WAV: {temp_wav}", flush=True)

    import ffmpeg
    print(f"[FASE 6] Exporting final MP3 via ffmpeg-python -> {out_mp3}...", flush=True)
    (
        ffmpeg
        .input(temp_wav)
        .output(out_mp3, audio_bitrate="320k", acodec="libmp3lame", ar=target_sr)
        .overwrite_output()
        .run(cmd=ffmpeg_exe, quiet=True)
    )

    if os.path.exists(out_mp3):
        print(f"[FASE 6] CAPOLAVORO FINALE COMPLETATO: {out_mp3} ({os.path.getsize(out_mp3)/(1024*1024):.2f} MB)", flush=True)
    else:
        print("ERROR: MP3 export failed!", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

import os
import sys

project_root = r"C:\Users\seren\Desktop\Progetto_Pandetta_Miku"
rvc_dir = os.path.join(project_root, "rvc_cli")
ffmpeg_dir = os.path.join(project_root, "venv", "Lib", "site-packages", "static_ffmpeg", "bin", "win32")
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

os.chdir(rvc_dir)
if rvc_dir not in sys.path:
    sys.path.insert(0, rvc_dir)

from core import run_infer_script

input_wav = os.path.join(project_root, "kaito_vocals.wav")
output_wav = os.path.join(project_root, "pandetta_vocals.wav")

pth_path = os.path.join(project_root, "models", "pandetta", "Niko_Pandetta_voce_QUALITY", "Niko Pandetta_400e_5600s.pth")
index_path = os.path.join(project_root, "models", "pandetta", "Niko_Pandetta_voce_QUALITY", "Niko Pandetta.index")

method = "rmvpe"
index_rate = 0.75
protect_voiceless = 0.33

print(f"[FASE 5] Starting RVC Inference on {input_wav}", flush=True)
print(f"  -> Model: {pth_path}", flush=True)
print(f"  -> Index: {index_path}", flush=True)
print(f"  -> Parameters: f0up_key=-12, method='{method}', index_rate={index_rate}, protect_voiceless={protect_voiceless}", flush=True)

msg, out_file = run_infer_script(
    pitch=-12,
    index_rate=index_rate,
    volume_envelope=1.0,
    protect=protect_voiceless,
    f0_method=method,
    input_path=input_wav,
    output_path=output_wav,
    pth_path=pth_path,
    index_path=index_path,
    f0_autotune=False,
    f0_autotune_strength=1.0,
    proposed_pitch=False,
    proposed_pitch_threshold=155.0,
    export_format="WAV",
    embedder_model="contentvec",
)

print(f"[FASE 5] {msg} -> {out_file}", flush=True)
if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
    print("ERROR: pandetta_vocals.wav was not generated!", flush=True)
    sys.exit(1)

print(f"[FASE 5] SUCCESS: pandetta_vocals.wav ({os.path.getsize(output_wav)/(1024*1024):.2f} MB)", flush=True)

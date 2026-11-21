import os
import shutil
import subprocess
import sys

ffmpeg_dir = r"C:\Users\seren\Desktop\Progetto_Pandetta_Miku\venv\Lib\site-packages\static_ffmpeg\bin\win32"
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

demucs_exe = r"C:\Users\seren\Desktop\Progetto_Pandetta_Miku\venv\Scripts\demucs.exe"

print("Running Demucs vocal separation (--two-stems=vocals) on miku_original.wav and kaito_original.wav...", flush=True)
cmd = [
    demucs_exe,
    "--two-stems=vocals",
    "-n", "htdemucs",
    "miku_original.wav",
    "kaito_original.wav"
]
subprocess.run(cmd, check=True)

# Find extracted vocals and move to root
mapping = {
    "miku_original": "miku_vocals.wav",
    "kaito_original": "kaito_vocals.wav"
}

for src_folder, target_name in mapping.items():
    found = False
    for root, dirs, files in os.walk("separated"):
        if os.path.basename(root) == src_folder and "vocals.wav" in files:
            src_path = os.path.join(root, "vocals.wav")
            shutil.copy2(src_path, target_name)
            print(f"Copied {src_path} -> {target_name} ({os.path.getsize(target_name)/(1024*1024):.2f} MB)", flush=True)
            found = True
            break
    if not found:
        print(f"ERROR: Could not find vocals.wav for {src_folder}", flush=True)
        sys.exit(1)

print("FASE 2 completed successfully!", flush=True)

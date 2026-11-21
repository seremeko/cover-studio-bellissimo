import os
import subprocess
import sys

ffmpeg_dir = r"C:\Users\seren\Desktop\Progetto_Pandetta_Miku\venv\Lib\site-packages\static_ffmpeg\bin\win32"
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

yt_dlp_exe = r"C:\Users\seren\Desktop\Progetto_Pandetta_Miku\venv\Scripts\yt-dlp.exe"

# Using matched 3:01 Cantarella ~Grace Edition~ tracks:
# - Miku solo cover (3:01): https://www.youtube.com/watch?v=JnLl2yYHcME
# - Kaito solo version (3:01): https://www.youtube.com/watch?v=u07JyFJh85w
# - Off-vocal / Instrumental (3:01): https://www.youtube.com/watch?v=wKT8yq0Jgis
targets = [
    ("https://www.youtube.com/watch?v=JnLl2yYHcME", "miku_original.wav"),
    ("https://www.youtube.com/watch?v=u07JyFJh85w", "kaito_original.wav"),
    ("https://www.youtube.com/watch?v=wKT8yq0Jgis", "instrumental.wav"),
]

for url, out_file in targets:
    if os.path.exists(out_file):
        os.remove(out_file)
    print(f"\n--- Downloading Cantarella: {url} -> {out_file} ---", flush=True)
    base_name = os.path.splitext(out_file)[0]
    cmd = [
        yt_dlp_exe,
        "--ffmpeg-location", ffmpeg_dir,
        "-x",
        "--audio-format", "wav",
        "-o", f"{base_name}.%(ext)s",
        url
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(out_file):
        size_mb = os.path.getsize(out_file) / (1024 * 1024)
        print(f"SUCCESS: {out_file} ({size_mb:.2f} MB)", flush=True)
    else:
        print(f"ERROR: {out_file} not found after download!", flush=True)
        sys.exit(1)

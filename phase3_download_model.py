import os
import zipfile
from huggingface_hub import hf_hub_download

os.makedirs("models/pandetta", exist_ok=True)

candidates = [
    ("Mikerx/NikoPandetta", "Niko Pandetta voce QUALITY.zip"),
    ("Mikerx/NikoPandetta", "nikopandetta.zip"),
    ("unitythe/NikoPandetta", "NikoPandetta_v1_250k.zip"),
]

for repo_id, filename in candidates:
    print(f"Downloading {repo_id} / {filename}...", flush=True)
    try:
        local_zip = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="models/pandetta_downloads"
        )
        print(f"Downloaded to {local_zip}", flush=True)
        extract_dir = os.path.join("models/pandetta", os.path.splitext(filename)[0].replace(" ", "_"))
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_zip, 'r') as zf:
            zf.extractall(extract_dir)
            print(f"Extracted contents of {filename}: {zf.namelist()}", flush=True)
    except Exception as e:
        print(f"Failed {repo_id}/{filename}: {e}", flush=True)

print("\nListing all .pth and .index files in models/pandetta:", flush=True)
for root, dirs, files in os.walk("models/pandetta"):
    for f in files:
        if f.endswith((".pth", ".index")):
            full_p = os.path.join(root, f)
            print(f"FOUND: {full_p} ({os.path.getsize(full_p)/(1024*1024):.2f} MB)", flush=True)

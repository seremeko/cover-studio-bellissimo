import os
import sys
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

queries = ["pandetta", "niko pandetta", "NikoPandetta", "Niko_Pandetta"]
found_files = []

print("Searching Hugging Face Hub Models & Datasets for 'pandetta'...", flush=True)

for q in queries:
    try:
        models = list(api.list_models(search=q, limit=50))
        for m in models:
            print(f"[MODEL REPO] {m.id}", flush=True)
            try:
                files = api.list_repo_files(repo_id=m.id, repo_type="model")
                for f in files:
                    if f.endswith((".pth", ".zip", ".index")):
                        print(f"   -> File: {f}", flush=True)
                        found_files.append(("model", m.id, f))
            except Exception as e:
                print(f"   -> Could not list files: {e}", flush=True)
    except Exception as e:
        print(f"Model search error for {q}: {e}", flush=True)

    try:
        datasets = list(api.list_datasets(search=q, limit=50))
        for d in datasets:
            print(f"[DATASET REPO] {d.id}", flush=True)
            try:
                files = api.list_repo_files(repo_id=d.id, repo_type="dataset")
                for f in files:
                    if f.endswith((".pth", ".zip", ".index")):
                        print(f"   -> File: {f}", flush=True)
                        found_files.append(("dataset", d.id, f))
            except Exception as e:
                print(f"   -> Could not list files: {e}", flush=True)
    except Exception as e:
        print(f"Dataset search error for {q}: {e}", flush=True)

    try:
        spaces = list(api.list_spaces(search=q, limit=50))
        for s in spaces:
            print(f"[SPACE REPO] {s.id}", flush=True)
    except Exception as e:
        pass

print(f"\nTotal candidate files found in named repos: {len(found_files)}", flush=True)

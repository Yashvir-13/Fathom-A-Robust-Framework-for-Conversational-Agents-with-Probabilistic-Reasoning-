"""
Download the e5-large-v2 model manually into the HuggingFace cache.
Uses urllib with progress display, or falls back to wget.
Supports resuming interrupted downloads.
"""
import os
import subprocess
import sys
from pathlib import Path

MODEL_ID = "intfloat/e5-large-v2"
# Files needed for the model
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json",
    "modules.json",
    "sentence_bert_config.json",
]

HF_TOKEN = os.environ.get("HF_TOKEN", "")
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

def download_with_wget(url, dest, token=None):
    """Download using wget with resume support."""
    cmd = ["wget", "-c", "--progress=bar:force:noscroll", "-O", str(dest)]
    if token:
        cmd.extend(["--header", f"Authorization: Bearer {token}"])
    cmd.append(url)
    subprocess.run(cmd, check=True)

def main():
    print(f"Downloading model: {MODEL_ID}")
    print(f"Token: {'set ✓' if HF_TOKEN else 'NOT SET ✗'}")
    
    # Create temp download directory
    dest_dir = Path("./model_cache/e5-large-v2")
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "1_Pooling").mkdir(exist_ok=True)
    
    base_url = f"https://huggingface.co/{MODEL_ID}/resolve/main"
    
    for filename in FILES:
        url = f"{base_url}/{filename}"
        dest = dest_dir / filename
        
        if dest.exists() and dest.stat().st_size > 0:
            # Skip small config files if already downloaded
            if "safetensors" not in filename:
                print(f"  ✓ {filename} (already exists)")
                continue
        
        print(f"\n  Downloading {filename}...")
        try:
            download_with_wget(url, dest, HF_TOKEN)
            print(f"  ✓ {filename} done")
        except Exception as e:
            print(f"  ✗ {filename} failed: {e}")
            return 1
    
    print(f"\n✅ Model downloaded to: {dest_dir.resolve()}")
    print(f"\nTo use it, update EMBEDDING_MODEL in evidence_gatherer.py to:")
    print(f'  EMBEDDING_MODEL = "{dest_dir.resolve()}"')
    return 0

if __name__ == "__main__":
    sys.exit(main())

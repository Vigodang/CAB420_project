import gdown
import zipfile
import os
from tqdm import tqdm

# ── Paste your file ID here ──────────────────────────────────────────────────
file_id = "1q6XGDflouIdgYpqlgLSAmoJZttr-HeVV"

# ── 1. Download from Google Drive ────────────────────────────────────────────
url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, "downloaded.zip", quiet=False)
print(f"✅ Saved: downloaded.zip ({os.path.getsize('downloaded.zip') / 1e6:.2f} MB)")

# ── 2. Extract with progress bar ─────────────────────────────────────────────
with zipfile.ZipFile("downloaded.zip", "r") as z:
    files = z.namelist()
    with tqdm(total=len(files), desc="Extracting", colour="blue") as bar:
        for file in files:
            z.extract(file, "./extracted_files")
            bar.update(1)

print(f"✅ Extracted {len(files)} files → ./extracted_files/")
print("\n📁 Contents:")
for f in os.listdir("./extracted_files"):
    print(f"   - {f}")
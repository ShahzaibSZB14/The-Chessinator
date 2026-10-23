"""Download the latest official Stockfish for Windows into ./engine/stockfish.exe"""
import io
import json
import os
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "engine", "stockfish.exe")
API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"


def main():
    with urllib.request.urlopen(API) as r:
        rel = json.load(r)
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    # prefer the fastest build that works everywhere on x86-64
    prefs = ["avx2", "universal", "sse41-popcnt", "x86-64.zip"]
    win = [n for n in assets if n.startswith("stockfish-windows-x86-64")]
    name = next((n for p in prefs for n in win if p in n), None)
    if not name:
        raise SystemExit(f"No Windows build found in release {rel['tag_name']}")
    print(f"Downloading {name} ({rel['tag_name']})...")
    with urllib.request.urlopen(assets[name]) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        exe = next(n for n in z.namelist() if n.endswith(".exe"))
        os.makedirs(os.path.dirname(DEST), exist_ok=True)
        with open(DEST, "wb") as f:
            f.write(z.read(exe))
    print(f"Saved to {DEST}")


if __name__ == "__main__":
    main()

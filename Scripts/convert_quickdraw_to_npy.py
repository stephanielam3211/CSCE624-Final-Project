import json
import numpy as np
from pathlib import Path

RAW_DIR = Path("../quickdraw_raw")
OUT_DIR = Path("../quickdraw_npy")
MAX_SAMPLES = 2000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for file in RAW_DIR.glob("*.ndjson"):
        drawings = []

        with open(file, "r") as f:
            for i, line in enumerate(f):
                if i >= MAX_SAMPLES:
                    break

                data = json.loads(line)

                if "recognized" in data and not data["recognized"]:
                    continue

                drawings.append(data["drawing"])

        out_file = OUT_DIR / (file.stem + ".npy")
        np.save(out_file, np.array(drawings, dtype=object))

        print(f"Saved {file.stem}")


if __name__ == "__main__":
    main()
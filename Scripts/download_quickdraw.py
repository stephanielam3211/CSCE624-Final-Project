from pathlib import Path
from urllib.request import urlretrieve

CLASSES = ["cat", "dog", "apple", "bicycle", "car", "fish", "tree", "house", "chair", "clock",]
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/raw"

OUT_DIR = Path("../quickdraw_raw")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        url = f"{BASE_URL}/{cls}.ndjson"
        out_file = OUT_DIR / f"{cls}.ndjson"

        if out_file.exists():
            print(f"Skipping {cls}")
            continue

        print(f"Downloading {cls}")
        urlretrieve(url, out_file)

    print("Done")


if __name__ == "__main__":
    main()
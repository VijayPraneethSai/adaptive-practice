import argparse
import subprocess
import zipfile
from pathlib import Path

COMPETITION = "riiid-test-answer-prediction"
FILES = ["train.csv", "questions.csv", "lectures.csv"]


def download(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{COMPETITION}.zip"
    if not zip_path.exists() and not all((out / f).exists() for f in FILES):
        subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(out)], check=True)
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
        zip_path.unlink()
    missing = [f for f in FILES if not (out / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing after download: {missing}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/riiid_raw")
    args = parser.parse_args()
    path = download(args.out)
    print(f"ready at {path}")


if __name__ == "__main__":
    main()

"""
prepare_data.py — Download and prepare the ISIC 2020 dataset

Supports:
  1. AUTO-DOWNLOAD from ISIC API (default)
  2. KAGGLE / CSV + flat images — train.csv + jpeg/train/ etc.
  3. HAM10000-style folders — Train/melanoma/, Train/nevus/, … (no CSV)

Usage examples:
  # Auto-download 5000 images from ISIC API:
  python prepare_data.py --n-images 5000 --data-dir data

  # Use your Kaggle dataset (recommended if you already have it):
  python prepare_data.py --kaggle-dir "C:/Users/you/Downloads/melanoma-classification"

  # Kaggle dir with custom output location:
  python prepare_data.py --kaggle-dir "/path/to/kaggle" --data-dir data --n-images 5000
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

METADATA_URL = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv"
ISIC_API_BASE = "https://api.isic-archive.com/api/v2"


# ─────────────────────────────────────────────────────────────────────────────
# KAGGLE MODE
# ─────────────────────────────────────────────────────────────────────────────

def find_kaggle_csv(kaggle_dir: Path) -> Path:
    candidates = [
        "train.csv",
        "ISIC_2020_Training_GroundTruth.csv",
        "train-labels.csv",
        "train_concat.csv",
        "train_labels.csv",
        "metadata.csv",
    ]
    for name in candidates:
        p = kaggle_dir / name
        if p.exists():
            return p
    exclude = {"sample_submission.csv", "test.csv"}
    csvs = sorted(
        [p for p in kaggle_dir.glob("*.csv") if p.name.lower() not in exclude],
        key=lambda p: p.name.lower(),
    )
    if not csvs:
        csvs = sorted(kaggle_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if csvs:
        preferred = [
            p
            for p in csvs
            if "train" in p.name.lower()
            or "ground" in p.name.lower()
            or "label" in p.name.lower()
        ]
        return preferred[0] if preferred else csvs[0]
    raise FileNotFoundError(
        f"No CSV found in {kaggle_dir}. "
        "Expected a file like train.csv or ISIC_2020_Training_GroundTruth.csv"
    )


def _dir_has_images(p: Path) -> bool:
    if not p.is_dir():
        return False
    for pat in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
        if any(p.glob(pat)):
            return True
    return False


def _count_images_in_dir(p: Path) -> int:
    n = 0
    for pat in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        n += len(list(p.glob(pat)))
    return n


def find_kaggle_image_dir(kaggle_dir: Path) -> Path:
    # SIIM-ISIC Kaggle layout: jpeg/train/ with .jpg tiles
    candidates = [
        kaggle_dir / "jpeg" / "train",
        kaggle_dir / "train",
        kaggle_dir / "train_images",
        kaggle_dir / "images",
        kaggle_dir / "jpeg" / "train_images",
    ]
    for p in candidates:
        if _dir_has_images(p):
            return p
    for p in sorted(kaggle_dir.rglob("*.jpg")):
        parent = p.parent
        cnt = _count_images_in_dir(parent)
        if cnt > 10:
            print(f"   Found images in: {parent} ({cnt} image files)")
            return parent
    for p in sorted(kaggle_dir.rglob("*.jpeg")):
        parent = p.parent
        cnt = _count_images_in_dir(parent)
        if cnt > 10:
            print(f"   Found images in: {parent} ({cnt} image files)")
            return parent
    raise FileNotFoundError(
        f"No image directory found under {kaggle_dir}. "
        "Make sure you extracted the Kaggle zip fully (expect e.g. jpeg/train/ with .jpg files)."
    )


def resolve_kaggle_image_path(image_dir: Path, stem: str) -> Optional[Path]:
    """Return path to an existing image file for this ISIC id (any common extension)."""
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        p = image_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def find_train_class_folder_root(kaggle_dir: Path) -> Optional[Path]:
    """
    Detect HAM10000-style layout: .../Train/<diagnosis>/*.jpg
    (e.g. 'Skin cancer ISIC.../Train/melanoma/ISIC_*.jpg').
    Returns the path to the Train directory, or None.
    """
    roots_to_try = [kaggle_dir.resolve()]
    try:
        for sub in kaggle_dir.iterdir():
            if sub.is_dir():
                roots_to_try.append(sub)
    except OSError:
        pass

    for base in roots_to_try:
        train_a = base / "Train"
        if (train_a / "melanoma").is_dir() and _dir_has_images(train_a / "melanoma"):
            return train_a
        # Already pointing at Train/
        if base.name.lower() == "train" and (base / "melanoma").is_dir():
            return base
    return None


def prepare_from_ham_class_folders(train_root: Path, data_dir, n_images, val_split, seed):
    """Build binary labels from folder names: melanoma=1, all other classes=0."""
    rows = []
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = 1 if class_dir.name.lower() == "melanoma" else 0
        for pat in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
            for img in class_dir.glob(pat):
                rows.append({"image_name": img.stem, "target": label, "_src": img})

    if not rows:
        raise RuntimeError(f"No images found under {train_root}")

    df = pd.DataFrame(rows)
    n_before = len(df)
    # Same ISIC id can appear in multiple folders; prefer melanoma (target=1) if conflict
    df = df.sort_values("target", ascending=False).drop_duplicates(subset=["image_name"], keep="first")
    dup = n_before - len(df)
    if dup:
        print(f"   Resolved {dup} duplicate image_name rows (kept melanoma label when present)")

    print(f"   Images indexed: {len(df):,}")
    print(
        f"   Class distribution:\n{df['target'].value_counts().rename({0: 'Benign (non-melanoma)', 1: 'Malignant (melanoma)'}).to_string()}"
    )

    malignant = df[df["target"] == 1]
    benign = df[df["target"] == 0]
    if len(malignant) == 0:
        raise RuntimeError("No melanoma images — expected a folder named 'melanoma' under Train/.")

    n_malignant = min(len(malignant), max(1, n_images // 4))
    n_benign = min(len(benign), max(0, n_images - n_malignant))

    sampled = pd.concat(
        [
            malignant.sample(n_malignant, random_state=seed),
            benign.sample(n_benign, random_state=seed),
        ]
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nSampled {len(sampled)} images:")
    print(f"   Benign (non-melanoma): {n_benign}")
    print(f"   Malignant (melanoma):  {n_malignant}")

    out_image_dir = data_dir / "images"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, row in sampled.iterrows():
        src = row["_src"]
        dest = out_image_dir / f"{row['image_name']}.jpg"
        if not dest.exists():
            if src.suffix.lower() in (".jpg", ".jpeg", ".jpe"):
                shutil.copy2(src, dest)
            else:
                Image.open(src).convert("RGB").save(dest, quality=95)
            copied += 1
    print(f"   Copied {copied} new images -> {out_image_dir}")

    _split_and_save(sampled[["image_name", "target"]], data_dir, val_split, seed)


def find_kaggle_csv_and_images(kaggle_dir: Path) -> Tuple[Path, Path]:
    """
    Locate train CSV and image folder. Tries the given path, then each direct
    subdirectory (Kaggle zips often unpack to one inner folder).
    """
    candidates = [kaggle_dir]
    try:
        for sub in sorted(kaggle_dir.iterdir()):
            if sub.is_dir():
                candidates.append(sub)
    except OSError:
        pass

    last_error: Optional[Exception] = None
    for base in candidates:
        try:
            csv_path = find_kaggle_csv(base)
            image_dir = find_kaggle_image_dir(base)
            if base != kaggle_dir:
                print(f"   Detected nested dataset folder: {base}")
            return csv_path, image_dir
        except FileNotFoundError as e:
            last_error = e
    raise FileNotFoundError(
        f"Could not find train CSV + images under {kaggle_dir}. "
        "Point --kaggle-dir at the folder that contains train.csv and jpeg/train (or train/). "
        f"Last error: {last_error}"
    )


def normalise_kaggle_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    img_col_map = {"image_name": "image_name", "image": "image_name",
                   "isic_id": "image_name", "id": "image_name", "filename": "image_name"}
    for old, new in img_col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
            break
    if "image_name" not in df.columns:
        for col in df.columns:
            if df[col].dtype == object and df[col].str.startswith("ISIC_").any():
                df = df.rename(columns={col: "image_name"})
                break
    if "image_name" in df.columns:
        df["image_name"] = df["image_name"].str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)

    label_col_map = {"target": "target", "label": "target", "class": "target"}
    for old, new in label_col_map.items():
        if old in df.columns and old != "target":
            df = df.rename(columns={old: new})
            break
    if "target" not in df.columns and "benign_malignant" in df.columns:
        df["target"] = (df["benign_malignant"].str.lower() == "malignant").astype(int)
    if "target" not in df.columns and "diagnosis" in df.columns:
        df["target"] = (df["diagnosis"].str.lower() == "melanoma").astype(int)
    if "target" in df.columns:
        df["target"] = df["target"].astype(int)

    missing = [c for c in ["image_name", "target"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find columns {missing} in the CSV. "
            f"Available columns: {list(df.columns)}"
        )
    return df[["image_name", "target"]]


def prepare_from_kaggle(kaggle_dir, data_dir, n_images, val_split, seed):
    print(f"\nUsing local dataset from: {kaggle_dir}")
    train_class_root = find_train_class_folder_root(kaggle_dir)
    if train_class_root is not None:
        print(f"   CSV + flat folder not used; using class folders under: {train_class_root}")
        prepare_from_ham_class_folders(train_class_root, data_dir, n_images, val_split, seed)
        return

    csv_path, image_dir = find_kaggle_csv_and_images(kaggle_dir)
    print(f"   CSV:    {csv_path.name}")
    print(f"   Images: {image_dir}")

    df = pd.read_csv(csv_path)
    print(f"   Rows in CSV: {len(df):,} | Columns: {list(df.columns)}")
    df = normalise_kaggle_df(df)
    print(f"   After normalisation: {len(df):,} rows")
    print(f"   Class distribution:\n{df['target'].value_counts().rename({0: 'Benign', 1: 'Malignant'}).to_string()}")

    def _has_image(n: str) -> bool:
        return resolve_kaggle_image_path(image_dir, n) is not None

    df = df[df["image_name"].apply(_has_image)].reset_index(drop=True)
    print(f"\n   Images found on disk: {len(df):,}")

    if len(df) == 0:
        raise RuntimeError(
            "No images found on disk matching the CSV entries. "
            "Double-check --kaggle-dir points to the extracted folder with images."
        )

    malignant = df[df["target"] == 1]
    benign    = df[df["target"] == 0]
    n_malignant = min(len(malignant), max(1, n_images // 4))
    n_benign    = min(len(benign), max(0, n_images - n_malignant))

    sampled = pd.concat([
        malignant.sample(n_malignant, random_state=seed),
        benign.sample(n_benign,       random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nSampled {len(sampled)} images:")
    print(f"   Benign:    {n_benign}")
    print(f"   Malignant: {n_malignant}")

    out_image_dir = data_dir / "images"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for name in sampled["image_name"]:
        src = resolve_kaggle_image_path(image_dir, name)
        if src is None:
            continue
        dest = out_image_dir / f"{name}.jpg"
        if not dest.exists():
            try:
                os.symlink(src.resolve(), dest)
            except (OSError, NotImplementedError):
                if src.suffix.lower() in (".jpg", ".jpeg", ".jpe"):
                    shutil.copy2(src, dest)
                else:
                    Image.open(src).convert("RGB").save(dest, quality=95)
            linked += 1
    print(f"   Linked/copied {linked} new images -> {out_image_dir}")
    _split_and_save(sampled, data_dir, val_split, seed)


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DOWNLOAD MODE
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url, dest, desc="Downloading"):
    import requests
    from tqdm import tqdm
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_isic_images(image_names, image_dir, max_workers=4):
    import requests
    import concurrent.futures
    from tqdm import tqdm
    image_dir.mkdir(parents=True, exist_ok=True)

    def fetch_one(name):
        dest = image_dir / f"{name}.jpg"
        if dest.exists():
            return True
        try:
            r = requests.get(f"{ISIC_API_BASE}/images/{name}/download/", timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except Exception:
            return False

    print(f"\nDownloading {len(image_names)} images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(fetch_one, image_names), total=len(image_names), desc="Images"))

    success = sum(results)
    print(f"Downloaded {success}/{len(image_names)} images")
    if success < len(image_names):
        print(f"Warning: {len(image_names) - success} failed; they will be skipped")
    return success


def prepare_from_download(n_images, data_dir, val_split, seed, workers):
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir = data_dir / "images"

    print("Downloading ISIC 2020 metadata...")
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        download_file(METADATA_URL, meta_path, desc="Metadata CSV")

    df = pd.read_csv(meta_path)
    print(f"   Total images in dataset: {len(df):,}")
    print(f"   Class distribution:\n{df['target'].value_counts().to_string()}")

    malignant   = df[df["target"] == 1]
    benign      = df[df["target"] == 0]
    n_malignant = min(len(malignant), n_images // 4)
    n_benign    = min(len(benign),    n_images - n_malignant)

    sampled = pd.concat([
        malignant.sample(n_malignant, random_state=seed),
        benign.sample(n_benign,       random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nSampled {len(sampled)} images | Benign: {n_benign} | Malignant: {n_malignant}")
    download_isic_images(sampled["image_name"].tolist(), image_dir, max_workers=workers)

    sampled = sampled[
        sampled["image_name"].apply(lambda x: (image_dir / f"{x}.jpg").exists())
    ].reset_index(drop=True)
    print(f"   {len(sampled)} images available after download")

    if len(sampled) < 10:
        raise RuntimeError(
            f"Only {len(sampled)} images downloaded. "
            "Check your internet connection or use --kaggle-dir instead."
        )

    _split_and_save(sampled, data_dir, val_split, seed)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_and_save(sampled, data_dir, val_split, seed):
    if len(sampled) < 2:
        raise RuntimeError(f"Need at least 2 samples to split, got {len(sampled)}.")

    class_counts = sampled["target"].value_counts()
    use_stratify = (class_counts >= 2).all()
    if not use_stratify:
        print("Warning: not enough samples per class for stratified split; using random split.")

    train_df, val_df = train_test_split(
        sampled,
        test_size=val_split,
        stratify=sampled["target"] if use_stratify else None,
        random_state=seed,
    )

    train_csv = data_dir / "train.csv"
    val_csv   = data_dir / "val.csv"
    train_df[["image_name", "target"]].to_csv(train_csv, index=False)
    val_df[["image_name", "target"]].to_csv(val_csv,   index=False)

    print(f"\nDataset ready.")
    print(f"   Train: {len(train_df)} images -> {train_csv}")
    print(f"   Val:   {len(val_df)} images -> {val_csv}")
    print(f"   Images: {data_dir / 'images'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ISIC dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py --n-images 5000
  python prepare_data.py --kaggle-dir "C:/Users/you/Downloads/melanoma-classification"
  python prepare_data.py --kaggle-dir /path/to/kaggle --n-images 3000 --data-dir data
        """,
    )
    parser.add_argument("--n-images",  type=int,   default=5000, help="Total images to use")
    parser.add_argument("--data-dir",  default="data",           help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.2,  help="Validation fraction")
    parser.add_argument("--workers",   type=int,   default=4,    help="Download worker threads")
    parser.add_argument(
        "--kaggle-dir", default=None,
        help="Path to extracted Kaggle ISIC dataset (skips downloading)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.kaggle_dir:
        kaggle_dir = Path(args.kaggle_dir)
        if not kaggle_dir.exists():
            raise FileNotFoundError(f"--kaggle-dir not found: {kaggle_dir}")
        prepare_from_kaggle(
            kaggle_dir=kaggle_dir,
            data_dir=data_dir,
            n_images=args.n_images,
            val_split=args.val_split,
            seed=42,
        )
    else:
        prepare_from_download(
            n_images=args.n_images,
            data_dir=data_dir,
            val_split=args.val_split,
            seed=42,
            workers=args.workers,
        )

    print("\nNext step (from project root):")
    print("   python src/train.py")

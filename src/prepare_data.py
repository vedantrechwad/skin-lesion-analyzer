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

# Classes considered malignant/pre-cancerous for binary mapping
MALIGNANT_CLASSES = {"melanoma", "basal cell carcinoma", "squamous cell carcinoma", "actinic keratosis"}


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


def _classify_class_folder(folder_name: str) -> int:
    """Map a class folder name to binary label: malignant/pre-cancerous=1, benign=0."""
    return 1 if folder_name.lower() in MALIGNANT_CLASSES else 0


def _index_class_folders(root: Path) -> pd.DataFrame:
    """
    Scan class-name sub-directories under `root` and build a DataFrame with
    columns: image_name, target, class_label, _src. Uses MALIGNANT_CLASSES for labelling.
    """
    rows = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = _classify_class_folder(class_dir.name)
        for pat in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
            for img in class_dir.glob(pat):
                rows.append({
                    "image_name": img.stem, 
                    "target": label, 
                    "class_label": class_dir.name.lower(),
                    "_src": img
                })
    if not rows:
        raise RuntimeError(f"No images found under {root}")
    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df.sort_values("target", ascending=False).drop_duplicates(subset=["image_name"], keep="first")
    dup = n_before - len(df)
    if dup:
        print(f"   Resolved {dup} duplicate image_name rows (kept malignant label when present)")
    return df


def prepare_from_ham_class_folders(train_root: Path, data_dir, n_images, val_split, seed):
    """Build binary labels from folder names using MALIGNANT_CLASSES mapping."""
    df = _index_class_folders(train_root)

    print(f"   Images indexed: {len(df):,}")
    print(
        f"   Class distribution:\n{df['target'].value_counts().rename({0: 'Benign', 1: 'Malignant'}).to_string()}"
    )

    malignant = df[df["target"] == 1]
    benign = df[df["target"] == 0]
    if len(malignant) == 0:
        raise RuntimeError("No malignant images found — expected folders matching MALIGNANT_CLASSES under Train/.")

    n_malignant = min(len(malignant), max(1, n_images // 4))
    n_benign = min(len(benign), max(0, n_images - n_malignant))

    sampled = pd.concat(
        [
            malignant.sample(n_malignant, random_state=seed),
            benign.sample(n_benign, random_state=seed),
        ]
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nSampled {len(sampled)} images:")
    print(f"   Benign:    {n_benign}")
    print(f"   Malignant: {n_malignant}")

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


# ─────────────────────────────────────────────────────────────────────────────
# ISIC MERGE MODE — merge class-folder dataset with existing data
# ─────────────────────────────────────────────────────────────────────────────

def _copy_images_to_dir(df: pd.DataFrame, out_image_dir: Path) -> int:
    """Copy images listed in df (with '_src' column) into out_image_dir. Returns count of new copies."""
    out_image_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, row in df.iterrows():
        src = Path(row["_src"])
        dest = out_image_dir / f"{row['image_name']}.jpg"
        if not dest.exists():
            if src.suffix.lower() in (".jpg", ".jpeg", ".jpe"):
                shutil.copy2(src, dest)
            else:
                Image.open(src).convert("RGB").save(dest, quality=95)
            copied += 1
    return copied


def prepare_and_merge_isic(isic_dir: Path, data_dir: Path, val_split: float, seed: int, generate_multiclass: bool = False):
    """
    Merge an ISIC class-folder dataset (Train/ and optionally Test/) with
    existing data in data_dir.  Uses MALIGNANT_CLASSES for binary labelling.
    All images are used (no sub-sampling) to maximise training data.
    If generate_multiclass is True, also generates train_multiclass.csv 
    and val_multiclass.csv.
    """
    print(f"\n{'='*60}")
    print(f"  ISIC MERGE MODE")
    print(f"  Source: {isic_dir}")
    print(f"  Target: {data_dir}")
    print(f"  Multiclass CSVs: {'Yes' if generate_multiclass else 'No'}")
    print(f"{'='*60}")

    out_image_dir = data_dir / "images"

    # ── Index new images from Train/ and Test/ ────────────────────────────
    new_frames = []

    train_root = isic_dir / "Train"
    if train_root.is_dir():
        print(f"\nScanning Train folder: {train_root}")
        train_df = _index_class_folders(train_root)
        print(f"   Train images indexed: {len(train_df):,}")
        print(f"   Distribution:\n{train_df['target'].value_counts().rename({0: 'Benign', 1: 'Malignant'}).to_string()}")
        new_frames.append(train_df)

    test_root = isic_dir / "Test"
    if test_root.is_dir():
        print(f"\nScanning Test folder: {test_root}")
        test_df = _index_class_folders(test_root)
        print(f"   Test images indexed: {len(test_df):,}")
        new_frames.append(test_df)

    if not new_frames:
        raise RuntimeError(f"No Train/ or Test/ folder found under {isic_dir}")

    new_df = pd.concat(new_frames, ignore_index=True)
    new_df = new_df.sort_values("target", ascending=False).drop_duplicates(subset=["image_name"], keep="first")
    print(f"\nTotal new images to add: {len(new_df):,}")
    print(f"   Malignant: {(new_df['target'] == 1).sum()}")
    print(f"   Benign:    {(new_df['target'] == 0).sum()}")

    # ── Copy new images to data/images/ ───────────────────────────────────
    copied = _copy_images_to_dir(new_df, out_image_dir)
    print(f"   Copied {copied} new images -> {out_image_dir}")

    # ── Load existing CSVs and merge ──────────────────────────────────────
    existing_train_csv = data_dir / "train.csv"
    existing_val_csv   = data_dir / "val.csv"
    existing_frames = []

    if existing_train_csv.exists():
        et = pd.read_csv(existing_train_csv)
        print(f"\nExisting train.csv: {len(et)} rows")
        existing_frames.append(et[["image_name", "target"]])
    if existing_val_csv.exists():
        ev = pd.read_csv(existing_val_csv)
        print(f"Existing val.csv:   {len(ev)} rows")
        existing_frames.append(ev[["image_name", "target"]])

    # Merge: combine existing + new, deduplicate (prefer malignant label)
    # Be sure to include class_label from new_df when available
    all_frames = existing_frames + [new_df]
    merged = pd.concat(all_frames, ignore_index=True)
    n_before = len(merged)
    # Sort by class_label first (so non-null comes up), then by target so malignant is kept
    merged = merged.sort_values(["target", "class_label"], ascending=[False, False], na_position='last')
    merged = merged.drop_duplicates(subset=["image_name"], keep="first")
    n_after = len(merged)
    if n_before != n_after:
        print(f"   Deduplicated: {n_before} -> {n_after} ({n_before - n_after} overlaps resolved)")

    # Verify images exist on disk
    merged = merged[merged["image_name"].apply(
        lambda x: (out_image_dir / f"{x}.jpg").exists()
    )].reset_index(drop=True)
    print(f"\nFinal merged dataset: {len(merged)} images")
    print(f"   Malignant: {(merged['target'] == 1).sum()}")
    print(f"   Benign:    {(merged['target'] == 0).sum()}")

    # ── Back up old CSVs, then split and save ─────────────────────────────
    for csv_path in (existing_train_csv, existing_val_csv):
        if csv_path.exists():
            backup = csv_path.with_suffix(".csv.bak")
            shutil.copy2(csv_path, backup)
            print(f"   Backed up {csv_path.name} -> {backup.name}")

    _split_and_save(merged, data_dir, val_split, seed)

    # ── Optionally split and save for multiclass ────────────────────────────
    if generate_multiclass and "class_label" in merged.columns:
        print("\n   Generating multiclass CSVs...")
        # For existing datasets that didn't have class_label, it will be NaN.
        # We drop those rows. The multi-class model will only train on the ISIC subset.
        mc_merged = merged.dropna(subset=["class_label"]).copy()
        
        class_counts = mc_merged["class_label"].value_counts()
        use_stratify = (class_counts >= 2).all()
        
        mc_train, mc_val = train_test_split(
            mc_merged,
            test_size=val_split,
            stratify=mc_merged["class_label"] if use_stratify else None,
            random_state=seed,
        )
        
        mc_train_csv = data_dir / "train_multiclass.csv"
        mc_val_csv   = data_dir / "val_multiclass.csv"
        mc_train[["image_name", "class_label"]].to_csv(mc_train_csv, index=False)
        mc_val[["image_name", "class_label"]].to_csv(mc_val_csv, index=False)
        
        print(f"   Multiclass Train: {len(mc_train)} images -> {mc_train_csv}")
        print(f"   Multiclass Val:   {len(mc_val)} images -> {mc_val_csv}")
        print(f"   (Excluded {len(merged) - len(mc_merged)} legacy images without class labels)")


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
  python prepare_data.py --isic-dir "Skin cancer ISIC The International Skin Imaging Collaboration"
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
    parser.add_argument(
        "--isic-dir", default=None,
        help="Path to ISIC class-folder dataset (Train/<class>/*.jpg). "
             "Merges with existing data in --data-dir.",
    )
    parser.add_argument(
        "--multiclass", action="store_true",
        help="If using --isic-dir, generate train/val_multiclass.csv alongside binary CSVs.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.isic_dir:
        isic_dir = Path(args.isic_dir)
        if not isic_dir.exists():
            raise FileNotFoundError(f"--isic-dir not found: {isic_dir}")
        prepare_and_merge_isic(
            isic_dir=isic_dir,
            data_dir=data_dir,
            val_split=args.val_split,
            seed=42,
            generate_multiclass=args.multiclass,
        )
    elif args.kaggle_dir:
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
    print("   python src/train_finetune.py   # if fine-tuning from existing model")

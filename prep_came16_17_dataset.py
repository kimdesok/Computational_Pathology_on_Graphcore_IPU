# prepare_came16_came17_features.py

import os
import re
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ======================================================================================
# CONFIG
# ======================================================================================
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

C16_LABELS = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/annotations/came16-labels.csv"
C17_LABELS = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/annotations/came17-labels.csv"

C16_PT = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/r50/pt_files"
C16_H5 = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/patches"
C17_PT = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/r50/pt_files"
C17_H5 = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/patches"

C16_XML = "/home/work/mytest/datasets/camelyon16/annotations"
C17_XML = "/home/work/mytest/datasets/camelyon17/annotations"

OUT_DIR = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/annotations/combined"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ======================================================================================
# HELPERS
# ======================================================================================
def read_csv_auto(path):
    return pd.read_csv(path, sep=None, engine="python")


def feature_tensor(payload):
    return payload["features"] if isinstance(payload, dict) else payload


def check_pair(pt_path, h5_path):
    payload = torch.load(pt_path, map_location="cpu")
    features = feature_tensor(payload)

    with h5py.File(h5_path, "r") as f:
        n_coords = len(f["coords"])

    if len(features) != n_coords:
        raise ValueError(f"{Path(pt_path).stem}: features={len(features)}, coords={n_coords}")

    return len(features), int(features.shape[-1])


def xml_path(xml_dir, slide_id):
    p = Path(xml_dir) / f"{slide_id}.xml"
    return str(p.resolve()) if p.exists() else ""


def local_paths(pt_dir, h5_dir):
    records = []

    for pt in sorted(Path(pt_dir).glob("*.pt")):
        sid = pt.stem
        h5 = Path(h5_dir) / f"{sid}.h5"

        if not h5.exists():
            print(f"WARNING: missing H5: {sid}")
            continue

        records.append({"slide_id": sid, "file_path": str(pt.resolve()), "coord_path": str(h5.resolve())})

    return pd.DataFrame(records)


# ======================================================================================
# CAMELYON16
# ======================================================================================
def build_camelyon16():
    meta = read_csv_auto(C16_LABELS)
    meta.columns = [c.strip().lower() for c in meta.columns]

    meta["image"] = meta["image"].astype(str).str.strip()
    meta["type"] = meta["type"].astype(str).str.strip().str.lower()
    meta["class"] = meta.get("class", "").astype(str).str.strip().str.lower()
    meta["slide_id"] = meta["image"].map(lambda x: Path(x).stem)

    def label(row):
        t, c = row["type"], row["class"]
        if t in {"tumor", "macro", "micro"} or c in {"tumor", "macro", "micro"}: return 1
        if t in {"normal", "negative"} or c in {"negative", "normal"}: return 0
        raise ValueError(f"Unknown Cam16 label: slide={row['slide_id']}, type={t}, class={c}")

    meta["label"] = meta.apply(label, axis=1)
    local = local_paths(C16_PT, C16_H5)
    df = local.merge(meta[["slide_id", "label", "type", "class", "center"]], on="slide_id", how="inner")

    df["dataset"] = "came16"
    df["patient_id"] = df["slide_id"]              # no patient linkage assumed
    df["group_id"] = "came16_" + df["patient_id"]
    df["xml_path"] = df["slide_id"].map(lambda x: xml_path(C16_XML, x))

    return df


# ======================================================================================
# CAMELYON17
# ======================================================================================
def build_camelyon17():
    meta = read_csv_auto(C17_LABELS)
    meta.columns = [c.strip().lower() for c in meta.columns]

    required = {"patient", "stage", "center"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Cam17 label CSV requires {required}; got {list(meta.columns)}")

    meta["patient"] = meta["patient"].astype(str).str.strip()
    meta["stage"] = meta["stage"].astype(str).str.strip().str.lower()

    # Remove patient-level rows such as patient_000.zip / pN0.
    meta = meta[meta["patient"].str.lower().str.endswith((".tif", ".tiff"))].copy()
    meta["slide_id"] = meta["patient"].map(lambda x: Path(x).stem)

    allowed = {"negative": 0, "itc": 0, "micro": 1, "macro": 1}
    unknown = sorted(set(meta["stage"]) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown Cam17 slide stages: {unknown}")

    meta["label"] = meta["stage"].map(allowed).astype(int)
    meta["class"] = meta["stage"]
    meta["type"] = np.where(meta["label"] == 1, "tumor", "normal")

    # patient_000_node_3 -> patient_000
    meta["patient_id"] = meta["slide_id"].str.extract(r"^(patient_\d+)_node_\d+$", expand=False)

    if meta["patient_id"].isna().any():
        bad = meta.loc[meta["patient_id"].isna(), "slide_id"].tolist()
        raise ValueError(f"Cannot extract patient IDs: {bad[:10]}")

    local = local_paths(C17_PT, C17_H5)
    df = local.merge(meta[["slide_id", "patient_id", "label", "type", "class", "center"]], on="slide_id", how="inner")

    df["dataset"] = "came17"
    df["group_id"] = "came17_" + df["patient_id"]
    df["xml_path"] = df["slide_id"].map(lambda x: xml_path(C17_XML, x))

    return df


# ======================================================================================
# PATIENT/GROUP-SAFE STRATIFIED SPLIT
# ======================================================================================
def make_split(df):
    # Patient/group positive if ANY constituent slide is positive.
    groups = df.groupby("group_id", as_index=False).agg(group_label=("label", "max"), dataset=("dataset", "first"))

    # Stratify jointly by dataset and group label where possible.
    groups["stratum"] = groups["dataset"] + "_" + groups["group_label"].astype(str)

    train_g, temp_g = train_test_split(
        groups, test_size=VAL_RATIO + TEST_RATIO, random_state=SEED, stratify=groups["stratum"]
    )

    temp_g = temp_g.copy()
    temp_g["stratum"] = temp_g["dataset"] + "_" + temp_g["group_label"].astype(str)

    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_g, test_g = train_test_split(
        temp_g, test_size=relative_test, random_state=SEED, stratify=temp_g["stratum"]
    )

    split_map = {g: "train" for g in train_g.group_id}
    split_map.update({g: "val" for g in val_g.group_id})
    split_map.update({g: "test" for g in test_g.group_id})

    out = df.copy()
    out["split"] = out["group_id"].map(split_map)

    return out


# ======================================================================================
# AUDIT
# ======================================================================================
def audit(df):
    print("\n" + "=" * 75)
    print("DATASET AUDIT")
    print("=" * 75)

    if df["slide_id"].duplicated().any():
        raise ValueError(f"Duplicate slide IDs: {df.loc[df.slide_id.duplicated(), 'slide_id'].tolist()[:10]}")

    # No group/patient may occur in >1 split.
    leakage = df.groupby("group_id")["split"].nunique()
    if (leakage > 1).any():
        raise RuntimeError(f"GROUP LEAKAGE: {leakage[leakage > 1].index.tolist()[:10]}")

    print("\nSlides by dataset/split/label:")
    print(df.groupby(["dataset", "split", "label"]).size())

    print("\nUnique groups/patients:")
    print(df.groupby(["dataset", "split"])["group_id"].nunique())

    print("\nCam17 stage distribution:")
    print(df[df.dataset == "came17"].groupby(["split", "class"]).size())

    print("\nCam17 center distribution:")
    print(df[df.dataset == "came17"].groupby(["split", "center"]).size())

    pos = df[df.label == 1]
    print(f"\nPositive slides             : {len(pos)}")
    print(f"Positive slides with XML    : {(pos.xml_path != '').sum()}")
    print(f"Positive slides without XML : {(pos.xml_path == '').sum()}")

    # XML absence on negatives is normal.
    neg_xml = df[(df.label == 0) & (df.xml_path != "")]
    print(f"Negative/ITC slides with XML: {len(neg_xml)}")

    print("\n✓ No patient/group leakage detected.")


# ======================================================================================
# OPTIONAL FEATURE/H5 AUDIT
# ======================================================================================
def audit_feature_shapes(df):
    dims = set()

    for i, row in df.iterrows():
        n, dim = check_pair(row.file_path, row.coord_path)
        dims.add(dim)

        if (i + 1) % 100 == 0:
            print(f"Checked {i + 1}/{len(df)} slides")

    if len(dims) != 1:
        raise ValueError(f"Different feature dimensions detected: {dims}")

    print(f"✓ All feature/H5 pairs match; feature_dim={next(iter(dims))}")


# ======================================================================================
# MAIN
# ======================================================================================
def main():
    c16, c17 = build_camelyon16(), build_camelyon17()

    print(f"Cam16 mapped slides: {len(c16)}")
    print(f"Cam17 mapped slides: {len(c17)}")

    c16.to_csv(Path(OUT_DIR) / "came16_labels_mapped.csv", index=False)
    c17.to_csv(Path(OUT_DIR) / "came17_labels_mapped.csv", index=False)

    combined = pd.concat([c16, c17], ignore_index=True)
    combined = make_split(combined)

    audit(combined)

    # Run once when creating a new feature dataset. It loads every .pt file.
    audit_feature_shapes(combined)

    combined.to_csv(Path(OUT_DIR) / "came16_came17_all.csv", index=False)

    for split in ["train", "val", "test"]:
        part = combined[combined["split"] == split].reset_index(drop=True)
        part.to_csv(Path(OUT_DIR) / f"came16_came17_{split}.csv", index=False)

    print("\nSaved:")
    for p in sorted(Path(OUT_DIR).glob("*.csv")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
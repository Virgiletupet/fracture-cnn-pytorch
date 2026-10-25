"""Data quality audit, run before any modeling.

Checks, in order:
1. metadata and files agree (every image exists, the folder matches the label);
2. every file decodes (FracAtlas v7 contains truncated JPEGs);
3. exact duplicates (same bytes, MD5) and near-duplicates (same picture re-encoded or resized);
4. label conflicts inside duplicate groups (label noise);
5. variables that correlate with the label and could become shortcuts (implants, image size).

Output: data/interim/audit.csv (one row per image) and data/interim/audit_report.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import INTERIM_DIR, META_COLS, RAW_DIR

# Two images are near-duplicates if their 64x64 grayscale thumbnails have a Pearson correlation >= 0.99.
# Perceptual hashes (dHash) do not work here: all X-rays look alike at 8x9 pixels (dark background,
# bright bone), which produces thousands of false matches.
NEAR_DUP_SIZE = 64
NEAR_DUP_CORR = 0.99


def load_metadata() -> pd.DataFrame:
    """dataset.csv plus the path of each image (the folder encodes the label)."""
    df = pd.read_csv(RAW_DIR / "dataset.csv")
    folder = np.where(df["fractured"] == 1, "Fractured", "Non_fractured")
    df["path"] = [str(RAW_DIR / "images" / f / name) for f, name in zip(folder, df["image_id"])]
    df["label"] = df["fractured"].astype(int)
    return df


def inspect_file(path: str) -> dict:
    """Size, color mode, MD5, whether the JPEG decodes completely, and a normalized thumbnail."""
    raw = Path(path).read_bytes()
    info = {"md5": hashlib.md5(raw).hexdigest(), "truncated": False}
    with Image.open(path) as im:
        info["width"], info["height"], info["mode"] = im.width, im.height, im.mode
        try:
            im.load()
        except OSError:            # "image file is truncated (n bytes not processed)"
            info["truncated"] = True
    if not info["truncated"]:
        with Image.open(path) as im:
            thumb = np.asarray(im.convert("L").resize((NEAR_DUP_SIZE, NEAR_DUP_SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
        thumb = thumb.ravel()
        info["thumb"] = (thumb - thumb.mean()) / (thumb.std() + 1e-6) / np.sqrt(thumb.size)
    return info


def near_duplicate_pairs(thumbs: np.ndarray, threshold: float = NEAR_DUP_CORR) -> list[tuple[int, int, float]]:
    """All pairs (i, j) whose thumbnails correlate above the threshold. Thumbnails are z-scored and
    divided by sqrt(d), so a dot product is exactly the Pearson correlation."""
    corr = thumbs @ thumbs.T
    i, j = np.triu_indices(len(thumbs), k=1)
    keep = corr[i, j] >= threshold
    return [(int(a), int(b), float(c)) for a, b, c in zip(i[keep], j[keep], corr[i, j][keep])]


def connected_groups(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Union-find: images linked by a duplicate relation, directly or through others, share a group id."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    roots = [find(x) for x in range(n)]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots])


def run_audit() -> pd.DataFrame:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata()
    lines = ["# Data quality audit (FracAtlas)", ""]

    # 1. Metadata and files agree
    files = list((RAW_DIR / "images").glob("*/*.jpg"))
    names = pd.Series([p.name for p in files])
    in_both = sorted(names[names.duplicated()])
    missing = (~df["image_id"].isin(set(names))).sum()
    lines += [f"- Images in dataset.csv: {len(df)} ({df['label'].sum()} fractured, {(1 - df['label']).sum()} not fractured, "
              f"{df['label'].mean():.1%} positive); files on disk: {len(files)}; missing files: {missing}",
              f"- Present in both folders: {', '.join(in_both) or 'none'}. dataset.csv marks them fractured: the labels "
              "were corrected but stale copies remain in Non_fractured. Labels are read from the CSV, never from folder names."]

    # 2. Decoding, sizes, color modes, hashes
    infos = [inspect_file(p) for p in df["path"]]
    for key in ["md5", "truncated", "width", "height", "mode"]:
        df[key] = [x[key] for x in infos]
    trunc = df[df["truncated"]]
    lines += [f"- Truncated JPEGs: {len(trunc)}, of which fractured: {trunc['label'].sum()} "
              f"(an artifact present in one class only can become a shortcut, so they are excluded)",
              f"- Color modes: {df['mode'].value_counts().to_dict()} (everything is converted to grayscale)"]

    # 3. Exact and near-duplicates -> groups
    md5_groups = df.groupby("md5").indices
    exact_pairs = [(int(g[0]), int(k)) for g in md5_groups.values() if len(g) > 1 for k in g[1:]]
    ok = np.flatnonzero(~df["truncated"].to_numpy())
    thumbs = np.stack([infos[k]["thumb"] for k in ok])
    near = [(int(ok[a]), int(ok[b]), c) for a, b, c in near_duplicate_pairs(thumbs)]
    df["group"] = connected_groups(len(df), exact_pairs + [(a, b) for a, b, _ in near])
    sizes = df.groupby("group")["image_id"].transform("size")
    lines += [f"- Exact duplicates (same MD5): {sum(len(g) > 1 for g in md5_groups.values())} groups",
              f"- Near-duplicate pairs (thumbnail correlation >= {NEAR_DUP_CORR}): {len(near)}",
              f"- Images that belong to a duplicate group: {(sizes > 1).sum()} in {df.loc[sizes > 1, 'group'].nunique()} groups "
              "(a group always stays in a single split)"]

    # 4. Label noise: one group, two labels
    conflict = df.groupby("group")["label"].transform("nunique") > 1
    df["label_conflict"] = conflict
    lines += [f"- Duplicate groups with contradictory labels: {df.loc[conflict, 'group'].nunique()} "
              f"({conflict.sum()} images, excluded: the true label is unknown)"]

    # 5. Potential shortcuts: variables strongly associated with the label
    hw = df.groupby("hardware")["label"].agg(["size", "mean"])
    df["shape"] = np.select([df["width"] > 1000, df["width"] > df["height"]], ["large", "landscape"], "portrait")
    shape = df.groupby("shape")["label"].agg(["size", "mean"])
    lines += ["", "## Possible shortcuts", "",
              f"- Orthopedic hardware: {int(hw.loc[1, 'size'])} images, {hw.loc[1, 'mean']:.0%} fractured "
              f"(vs {hw.loc[0, 'mean']:.0%} without). The model could detect metal instead of fractures: "
              "performance is reported separately without hardware.",
              "- Fracture rate by image format: " + ", ".join(f"{k} {v['mean']:.0%} (n={int(v['size'])})" for k, v in shape.iterrows())
              + ". Formats come from different devices; the letterbox resize keeps the format from being an obvious cue."]
    for col in ["hand", "leg", "hip", "shoulder", "mixed"]:
        sub = df[df[col] == 1]
        lines.append(f"- {col}: {len(sub)} images, {sub['label'].mean():.0%} fractured")

    df["exclude"] = df["truncated"] | df["label_conflict"]
    lines += ["", f"**Kept for modeling: {(~df['exclude']).sum()} images ({df.loc[~df['exclude'], 'label'].sum()} fractured).**"]

    cols = ["image_id", "path", "label", *META_COLS, "width", "height", "mode", "shape", "md5", "truncated", "group", "label_conflict", "exclude"]
    df[cols].to_csv(INTERIM_DIR / "audit.csv", index=False)
    pd.DataFrame(near, columns=["i", "j", "corr"]).assign(
        image_i=lambda d: df["image_id"].to_numpy()[d["i"]], image_j=lambda d: df["image_id"].to_numpy()[d["j"]]
    ).to_csv(INTERIM_DIR / "near_duplicates.csv", index=False)
    (INTERIM_DIR / "audit_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return df

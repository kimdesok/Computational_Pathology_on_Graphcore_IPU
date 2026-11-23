#!/usr/bin/env python
# coding: utf-8

"""
Camelyon16 ResNet50-feature training on Graphcore IPU + inference on RBLN NPU.

MODE = "train"   : IPU box, trains a Transformer/MIL head over precomputed patch features.
MODE = "compile" : NPU box, loads the .pt checkpoint, compiles with RBLN, evaluates test slides.

Expected mapped CSV columns:
    file_path, coord_path, slide_id, label
Optional:
    center

Each feature file is either:
    Tensor [num_patches, feature_dim]
or a dict containing:
    {"features": Tensor [num_patches, feature_dim]}

Each HDF5 coord file contains:
    coords [num_patches, 2]
"""

import gc
import json
import os
import time
import warnings
from collections import Counter

import math
from torch.utils.data import Dataset



import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset

from collections import defaultdict, deque

os.environ["POPTORCH_LOG_LEVEL"] = "ERR"
warnings.filterwarnings("ignore", category=UserWarning, message="positional args are being deprecated")

try:
    import poptorch
except ImportError:
    poptorch = None


# -----------------------------------------------------------------------------
# Feature dataset
# -----------------------------------------------------------------------------
class CamelyonPatchFeatureDataset(Dataset):
    """Dataset driven strictly by the mapped Camelyon16 feature CSV."""

    def __init__(self, mapped_csv_path: str, split: str = "train"):
        super().__init__()
        self.df = pd.read_csv(mapped_csv_path)
        self.split = split.lower()

        required_columns = {"file_path", "coord_path", "slide_id", "label"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Mapped CSV is missing columns: {missing}. "
                f"Rebuild {mapped_csv_path}."
            )

        if "split" not in self.df.columns:
            raise ValueError("Mapped CSV must contain a 'split' column.")

        if self.split in {"train", "val", "test"}:
            self.df = self.df[self.df["split"] == self.split].reset_index(drop=True)
        elif self.split == "all":
            pass
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        coord_path = row["coord_path"]
        label = int(row["label"])
        slide_id = str(row["slide_id"])

        payload = torch.load(file_path, map_location="cpu")
        features = payload["features"] if isinstance(payload, dict) else payload
        features = torch.as_tensor(features, dtype=torch.float32)

        if features.ndim != 2:
            raise ValueError(
                f"{slide_id}: expected features [N, D], got {tuple(features.shape)}"
            )

        with h5py.File(coord_path, "r") as f:
            coords = torch.from_numpy(f["coords"][:]).long()

        if features.shape[0] != coords.shape[0]:
            raise ValueError(
                f"{slide_id}: {features.shape[0]} features != "
                f"{coords.shape[0]} coords"
            )
         
        return {
            "features": features,
            "coords": coords,
            "label": torch.tensor(label, dtype=torch.long),
            "slide_id": slide_id,
            "center": row.get("center", -1),
        }
       
class x_CamelyonPatchFeatureDataset(Dataset):
    """Dataset driven strictly by the mapped Camelyon16 feature CSV."""

    def __init__(self, mapped_csv_path: str, split: str = "train"):
        super().__init__()
        self.df = pd.read_csv(mapped_csv_path)
        self.split = split.lower()

        required_columns = {"file_path", "coord_path", "slide_id", "label"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Mapped CSV is missing columns: {missing}. "
                f"Rebuild {mapped_csv_path}."
            )

        if self.split == "train":
            self.df = self.df[
                ~self.df["slide_id"].astype(str).str.startswith("test_")
            ].reset_index(drop=True)
        elif self.split == "test":
            self.df = self.df[
                self.df["slide_id"].astype(str).str.startswith("test_")
            ].reset_index(drop=True)
        elif self.split == "all":
            pass
        else:
            raise ValueError(
                f"Invalid split '{split}'. Choose from ['train', 'test', 'all']."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        coord_path = row["coord_path"]
        label = int(row["label"])
        slide_id = str(row["slide_id"])

        payload = torch.load(file_path, map_location="cpu")
        features = payload["features"] if isinstance(payload, dict) else payload
        features = torch.as_tensor(features, dtype=torch.float32)

        if features.ndim != 2:
            raise ValueError(
                f"{slide_id}: expected features [N, D], got {tuple(features.shape)}"
            )

        with h5py.File(coord_path, "r") as f:
            coords = torch.from_numpy(f["coords"][:]).long()

        if features.shape[0] != coords.shape[0]:
            raise ValueError(
                f"{slide_id}: {features.shape[0]} features != "
                f"{coords.shape[0]} coords"
            )
         
        return {
            "features": features,
            "coords": coords,
            "label": torch.tensor(label, dtype=torch.long),
            "slide_id": slide_id,
            "center": row.get("center", -1),
        }
       

class FixedFeatureBagDataset(Dataset):
    """
    Converts each variable-length slide into one fixed-size feature bag.

    Static [BAG_SIZE, FEATURE_DIM] tensors are important for PopTorch and RBLN.
    During training, patches are randomly sampled when a slide has > BAG_SIZE
    features. Validation/test use deterministic, evenly spaced sampling.

    If a slide has fewer than BAG_SIZE patches, feature indices are repeated
    instead of padding. This keeps every accelerator token valid and static.
    """

    def __init__(self, slide_dataset, bag_size=256, training=False, seed=42,
                 positive_csv=None, pos_ratio=0.10, patch_spacing=256,
                 cluster_gap_patches=2, context_radius_patches=3,
                 min_bags_per_cluster=1, max_bags_per_cluster=8,
                 max_bags_per_slide=16, neg_bags_per_slide=6):
        self.slide_dataset = slide_dataset
        self.bag_size = int(bag_size)
        self.training = training
        self.seed = seed
        self.epoch = 0

        self.pos_ratio = pos_ratio
        self.patch_spacing = patch_spacing
        self.cluster_gap = cluster_gap_patches * patch_spacing
        self.context_radius = context_radius_patches * patch_spacing
        self.min_bags_per_cluster = min_bags_per_cluster
        self.max_bags_per_cluster = max_bags_per_cluster
        self.max_bags_per_slide = max_bags_per_slide
        self.neg_bags_per_slide = neg_bags_per_slide

        # positive feature indices from our XML audit
        self.pos_lookup = defaultdict(set)

        if positive_csv is not None:
            pos_df = pd.read_csv(positive_csv)
            for slide_id, g in pos_df.groupby("slide_id"):
                self.pos_lookup[str(slide_id)] = set(g["feature_idx"].astype(int))

        self.bag_plan = []
        self.cluster_info = {}

        self._build_bag_plan()
        
        self.slide_ids = sorted({
            plan["slide_id"]
            for plan in self.bag_plan
        })

        self.slide_id_to_idx = {
            slide_id: idx
            for idx, slide_id in enumerate(self.slide_ids)
        }

        self.idx_to_slide_id = {
            idx: slide_id
            for slide_id, idx in self.slide_id_to_idx.items()
        }

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        
    # ------------------------------------------------------------------
    # Connected components among XML-positive coordinates
    # ------------------------------------------------------------------
    def _find_clusters(self, coords, pos_indices):

        pos_indices = np.asarray(sorted(pos_indices), dtype=np.int64)

        if len(pos_indices) == 0:
            return []

        pos_coords = coords[pos_indices].cpu().numpy()
        visited = np.zeros(len(pos_indices), dtype=bool)
        clusters = []

        for start in range(len(pos_indices)):

            if visited[start]:
                continue

            queue = deque([start])
            visited[start] = True
            members = []

            while queue:

                i = queue.popleft()
                members.append(pos_indices[i])

                dx = np.abs(pos_coords[:, 0] - pos_coords[i, 0])
                dy = np.abs(pos_coords[:, 1] - pos_coords[i, 1])

                # Chebyshev/grid neighborhood:
                # horizontal, vertical and diagonal neighbors are connected.
                neighbor_mask = (
                    (dx <= self.cluster_gap) &
                    (dy <= self.cluster_gap)
                )

                for j in np.flatnonzero(neighbor_mask):
                    if not visited[j]:
                        visited[j] = True
                        queue.append(j)

            clusters.append(np.asarray(members, dtype=np.int64))

        return clusters


    def _bags_for_cluster(self, n_positive):

        # deliberately sublinear:
        # 1-15 pos -> ~1 bag
        # 64 pos   -> ~2 bags
        # 256 pos  -> ~4 bags
        # 1024 pos -> ~8 bags
        n = int(np.ceil(np.sqrt(n_positive) / 4.0))

        return int(np.clip(
            n,
            self.min_bags_per_cluster,
            self.max_bags_per_cluster
        ))


    def _build_bag_plan(self):

        self.bag_plan.clear()
        self.cluster_info.clear()

        for slide_idx in range(len(self.slide_dataset)):

            sample = self.slide_dataset[slide_idx]
            label = int(sample["label"])
            slide_id = str(sample["slide_id"])

            # ----------------------------------------------------------
            # Negative slide: retain ordinary random bags.
            # Use the old fixed-style contribution.
            # ----------------------------------------------------------
            if label == 0:

                    for bag_no in range(self.neg_bags_per_slide):
                        self.bag_plan.append({
                            "slide_idx": slide_idx,
                            "slide_id": slide_id,
                            "label": 0,
                            "cluster_id": -1,
                            "bag_no": bag_no,
                        })

                    continue

            # ----------------------------------------------------------
            # Positive slide
            # ----------------------------------------------------------
            pos_indices = self.pos_lookup.get(slide_id, set())

            if len(pos_indices) == 0:
                for bag_no in range(self.min_bags_per_cluster):
                    self.bag_plan.append({
                        "slide_idx": slide_idx, "slide_id": slide_id, "label": label, "cluster_id": -1, "bag_no": bag_no,
                    })
                continue

            clusters = self._find_clusters(sample["coords"], pos_indices)

            # Largest clusters first if slide hits its cap.
            clusters = sorted(clusters, key=len, reverse=True)

            self.cluster_info[slide_id] = clusters

            slide_bags = 0

            for cluster_id, cluster in enumerate(clusters):

                n_bags = self._bags_for_cluster(len(cluster))

                for bag_no in range(n_bags):

                    if slide_bags >= self.max_bags_per_slide:
                        break

                    self.bag_plan.append({
                        "slide_idx": slide_idx,
                        "slide_id": slide_id,
                        "label": 1,
                        "cluster_id": cluster_id,
                        "cluster_indices": cluster,
                        "bag_no": bag_no,
                    })

                    slide_bags += 1

                if slide_bags >= self.max_bags_per_slide:
                    break

    def __len__(self):
        return len(self.bag_plan)

    def _sample(self, indices, n, rng):

        indices = np.asarray(indices, dtype=np.int64)

        if n <= 0:
            return np.empty(0, dtype=np.int64)

        if len(indices) == 0:
            return np.empty(0, dtype=np.int64)

        return rng.choice(indices, size=n, replace=(len(indices) < n))


    def _positive_bag(self, sample, plan, rng):

        coords = sample["coords"]
        cluster = np.asarray(plan["cluster_indices"], dtype=np.int64)

        # Guarantee this approximate tumor fraction.
        n_pos = max(1, int(round(self.bag_size * self.pos_ratio)))
        n_pos = min(n_pos, self.bag_size)

        chosen_pos = self._sample(cluster, n_pos, rng)

        # --------------------------------------------------------------
        # Cluster geometry
        # Recruit context surrounding the entire cluster.
        # --------------------------------------------------------------
        cluster_coords = coords[cluster].cpu().numpy()
        all_coords = coords.cpu().numpy()

        xmin = cluster_coords[:, 0].min() - self.context_radius
        xmax = cluster_coords[:, 0].max() + self.context_radius
        ymin = cluster_coords[:, 1].min() - self.context_radius
        ymax = cluster_coords[:, 1].max() + self.context_radius

        nearby = np.flatnonzero(
            (all_coords[:, 0] >= xmin) &
            (all_coords[:, 0] <= xmax) &
            (all_coords[:, 1] >= ymin) &
            (all_coords[:, 1] <= ymax)
        )

        # Don't deliberately select the confirmed positives again as context.
        all_positive = np.asarray(
            sorted(self.pos_lookup[plan["slide_id"]]),
            dtype=np.int64
        )

        nearby_context = np.setdiff1d(nearby, all_positive, assume_unique=False)

        remaining = self.bag_size - len(chosen_pos)
        chosen_context = self._sample(
            nearby_context,
            min(remaining, len(nearby_context)),
            rng
        )

        chosen = np.concatenate([chosen_pos, chosen_context])
        remaining = self.bag_size - len(chosen)

        # --------------------------------------------------------------
        # If the local neighborhood is too small, fill from the slide.
        # --------------------------------------------------------------
        if remaining > 0:

            all_idx = np.arange(len(coords), dtype=np.int64)
            available = np.setdiff1d(all_idx, chosen, assume_unique=False)

            fill = self._sample(available, remaining, rng)
            chosen = np.concatenate([chosen, fill])

        rng.shuffle(chosen)

        return chosen


    def _ordinary_bag(self, n, rng):

        indices = np.arange(n, dtype=np.int64)

        return self._sample(
            indices,
            self.bag_size,
            rng
        )


    def __getitem__(self, idx):

        plan = self.bag_plan[idx]
        sample = self.slide_dataset[plan["slide_idx"]]

        features = sample["features"]
        coords = sample["coords"]

        rng = np.random.default_rng(
            self.seed + idx + self.epoch * 1_000_003
        )

        if plan["label"] == 1 and "cluster_indices" in plan:
            chosen = self._positive_bag(sample, plan, rng)
        else:
            chosen = self._ordinary_bag(len(features), rng)

        chosen = torch.as_tensor(chosen, dtype=torch.long)

        bag_features = features[chosen].float()
        bag_coords = coords[chosen].float()

        # Keep your existing coordinate normalization.
        cmin = bag_coords.min(dim=0).values
        cmax = bag_coords.max(dim=0).values
        bag_coords = (bag_coords - cmin) / (cmax - cmin).clamp_min(1.0)

        mask = torch.ones(self.bag_size, dtype=torch.float32)
        
        slide_idx = torch.tensor(
            self.slide_id_to_idx[str(sample["slide_id"])],
            dtype=torch.long,
        )

        return (bag_features, bag_coords, mask, sample["label"], slide_idx)


def get_labels(dataset):
    """Return integer labels from a Dataset or Subset without loading features."""
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        return [int(base.df.iloc[i]["label"]) for i in dataset.indices]
    if hasattr(dataset, "df"):
        return dataset.df["label"].astype(int).tolist()
    return [int(dataset[i]["label"]) for i in range(len(dataset))]


def infer_feature_dim(dataset):
    if len(dataset) == 0:
        raise ValueError("Cannot infer feature dimension from an empty dataset.")
    sample = dataset[0]
    return int(sample["features"].shape[1])


# ----------------------------------------------------------------------
# Spatial clustering of XML-confirmed positive patches
# ----------------------------------------------------------------------

def cluster_positive_patches(rows, patch_stride=256, cluster_gap=1):
    """
    Connected components on the patch grid.

    Two positive patches belong to the same cluster when their grid
    positions are within `cluster_gap` cells in x/y.

    Returns:
        list of dicts:
            {
                "cluster_id": int,
                "feature_indices": np.ndarray,
                "coords": np.ndarray,
                "n_pos": int,
                "bbox": (xmin, ymin, xmax, ymax),
                "centroid": (cx, cy)
            }
    """

    if len(rows) == 0:
        return []

    rows = rows.reset_index(drop=True)

    coords = rows[["x", "y"]].to_numpy(dtype=np.int64)
    feature_indices = rows["feature_idx"].to_numpy(dtype=np.int64)

    gx = np.rint(coords[:, 0] / patch_stride).astype(int)
    gy = np.rint(coords[:, 1] / patch_stride).astype(int)

    grid_to_rows = {}

    for i, key in enumerate(zip(gx, gy)):
        grid_to_rows.setdefault(key, []).append(i)

    unvisited = set(range(len(rows)))
    clusters = []

    neighbor_offsets = [
        (dx, dy)
        for dx in range(-cluster_gap, cluster_gap + 1)
        for dy in range(-cluster_gap, cluster_gap + 1)
        if not (dx == 0 and dy == 0)
    ]

    while unvisited:

        start = unvisited.pop()
        component = [start]
        stack = [start]

        while stack:

            i = stack.pop()
            xg, yg = gx[i], gy[i]

            for dx, dy in neighbor_offsets:

                neighbor_key = (xg + dx, yg + dy)

                for j in grid_to_rows.get(neighbor_key, []):

                    if j in unvisited:
                        unvisited.remove(j)
                        stack.append(j)
                        component.append(j)

        component = np.asarray(component, dtype=int)

        c_coords = coords[component]
        c_indices = feature_indices[component]

        xmin = int(c_coords[:, 0].min())
        ymin = int(c_coords[:, 1].min())
        xmax = int(c_coords[:, 0].max())
        ymax = int(c_coords[:, 1].max())

        clusters.append({
            "cluster_id": len(clusters),
            "feature_indices": c_indices,
            "coords": c_coords,
            "n_pos": len(c_indices),
            "bbox": (xmin, ymin, xmax, ymax),
            "centroid": (
                float(c_coords[:, 0].mean()),
                float(c_coords[:, 1].mean()),
            ),
        })

    # Largest tumor focus first
    clusters.sort(
        key=lambda x: x["n_pos"],
        reverse=True
    )

    # Restore IDs after sorting
    for cluster_id, cluster in enumerate(clusters):
        cluster["cluster_id"] = cluster_id

    return clusters

# -----------------------------------------------------------------------------
# Transformer over ResNet50 feature tokens
# -----------------------------------------------------------------------------
class FeatureTransformerCamelyon(nn.Module):
    def __init__(self, class_weights, feature_dim=1024, embed_dim=768, num_heads=12,
                 num_layers=12, mlp_ratio=4, num_classes=2, dropout=0.0,
                 bag_size=256, num_ipus=4, use_coords=True):
        super().__init__()

        self.embed_dim, self.num_layers = embed_dim, num_layers
        self.num_ipus, self.use_coords = num_ipus, use_coords

        self.feature_proj = nn.Linear(feature_dim, embed_dim)

        if use_coords:
            self.coord_embed = nn.Sequential(
                nn.Linear(2, embed_dim), nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self._assign_ipus()


    def _assign_ipus(self):
        if self.num_ipus == 4:
            layer_to_ipu = [0,0,0, 1,1,1, 2,2,2, 3,3,3]

        elif self.num_ipus == 8:
            # Same 12-layer model; only distributed across 8 IPUs
            layer_to_ipu = [0, 1,1, 2, 3,3, 4, 5,5, 6, 7,7]

        else:
            raise ValueError("num_ipus must be 4 or 8")

        self.feature_proj = poptorch.BeginBlock(
            self.feature_proj, "feature_proj", ipu_id=0
        )

        if self.use_coords:
            self.coord_embed = poptorch.BeginBlock(
                self.coord_embed, "coord_embed", ipu_id=0
            )

        for i, ipu_id in enumerate(layer_to_ipu):
            self.blocks[i] = poptorch.BeginBlock(
                self.blocks[i], f"transformer_{i}", ipu_id=ipu_id
            )

        self.norm = poptorch.BeginBlock(
            self.norm, "final_norm", ipu_id=self.num_ipus - 1
        )

        self.classifier = poptorch.BeginBlock(
            self.classifier, "classifier", ipu_id=self.num_ipus - 1
        )


    def forward(self, features, coords, mask, labels=None):
        # features: [B, 256, 1024]
        # coords:   [B, 256, 2]

        x = self.feature_proj(features)

        if self.use_coords:
            coords = coords.to(dtype=x.dtype)
            x = x + self.coord_embed(coords)

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.classifier(x[:, 0])

        if labels is None:
            return logits

        labels = labels.long()

        log_probs = F.log_softmax(logits, dim=1)
        sample_loss = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        sample_weights = self.class_weights[labels].to(sample_loss.dtype)
        
        loss = (sample_loss * sample_weights).mean()
        loss = poptorch.identity_loss(loss, reduction="mean")
        
        return logits, loss
    
    
class x_FeatureTransformerCamelyon(nn.Module):
    """
    Slide classifier over precomputed patch features.

    Input:
        features : [batch, bag_size, feature_dim]
        coords   : [batch, bag_size, 2]
        mask     : [batch, bag_size], 1=real patch, 0=padding
    """

    def __init__(self, feature_dim=2048, bag_size=256, embed_dim=256, num_heads=4,
                 num_layers=4, mlp_ratio=4, dropout=0.1, num_classes=2,
                 use_coords=True, ipu_pipeline=False, pos_weight=None):

        super().__init__()

        self.feature_dim = int(feature_dim)
        self.bag_size = int(bag_size)
        self.embed_dim = int(embed_dim)
        self.use_coords = bool(use_coords)
        self.ipu_pipeline = bool(ipu_pipeline)
        self.pos_weight = 1.0 if pos_weight is None else float(pos_weight)

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.coord_proj = nn.Linear(2, embed_dim) if use_coords else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, bag_size + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()

        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


    def _stage(self, x, start, end, ipu_id):

        if self.ipu_pipeline and poptorch is not None:
            with poptorch.Block(f"Transformer_{start}_{end-1}", ipu_id=ipu_id):
                for i in range(start, end):
                    x = self.blocks[i](x)
                return x

        for i in range(start, end):
            x = self.blocks[i](x)

        return x


    def weighted_cross_entropy(self, logits, labels):

        log_probs = F.log_softmax(logits, dim=1)
        nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        weights = 1.0 + labels.to(logits.dtype) * (self.pos_weight - 1.0)
        weighted_loss = (nll * weights).sum() / weights.sum()
        return weighted_loss


    def forward(self, features, coords, mask, labels=None):

        # ----------------------------------------------------------
        # Feature projection
        # ----------------------------------------------------------
        if self.ipu_pipeline and poptorch is not None:
            with poptorch.Block("Feature_projection", ipu_id=0):
                x = self.feature_proj(features)
                if self.coord_proj is not None:
                    x = x + self.coord_proj(coords)
                x = x * mask.unsqueeze(-1)
                cls = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = self.pos_drop(
                    x + self.pos_embed[:, :x.shape[1]]
                )

        else:
            x = self.feature_proj(features)
            if self.coord_proj is not None:
                x = x + self.coord_proj(coords)
            x = x * mask.unsqueeze(-1)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.pos_drop(
                x + self.pos_embed[:, :x.shape[1]]
            )

        # Transformer stages
        n_layers = len(self.blocks)

        boundaries = [
            0,
            min(1, n_layers),
            min(2, n_layers),
            min(3, n_layers),
            n_layers,
        ]

        x = self._stage(x, boundaries[0], boundaries[1], 0)
        x = self._stage(x, boundaries[1], boundaries[2], 1)
        x = self._stage(x, boundaries[2], boundaries[3], 2)
        x = self._stage(x, boundaries[3], boundaries[4], 3)

        # IPU head + weighted loss
 
        if self.ipu_pipeline and poptorch is not None:
            with poptorch.Block("Head", ipu_id=3):
                logits = self.head(
                    self.norm(x[:, 0])
                )
                if labels is not None:
                    weighted_loss = self.weighted_cross_entropy(
                        logits,
                        labels,
                    )
                    loss = poptorch.identity_loss(
                        weighted_loss,
                        reduction="none",
                    )
                    return logits, loss
                return logits

        # Normal PyTorch / CPU path
        logits = self.head(
            self.norm(x[:, 0])
        )

        if labels is not None:
            loss = self.weighted_cross_entropy(
                logits,
                labels,
            )
            return logits, loss
        return logits

# Shared helpers
def cleanup_poptorch(*models):
    for name, obj in models:
        if obj is None:
            continue
        try:
            obj.detachFromDevice()
        except Exception:
            pass
        try:
            obj.destroy()
        except Exception:
            pass
    gc.collect()

def evaluate_ipu(poptorch_infer_model, data_loader, verbose=False):
    total = 0
    correct = 0
    total_loss = 0.0
    n_steps = 0

    pred_counter = Counter()
    label_counter = Counter()

    with torch.no_grad():
        for (features, coords, mask, labels, slide_indices) in data_loader:
            features = features.half()
            coords = coords.half()
            mask = mask.half()

            logits = poptorch_infer_model(
                features, coords, mask
            ).float()

            loss = weighted_cross_entropy_eval(
                logits, labels, pos_weight=1.0,
            )

            preds = torch.argmax(logits, dim=1)

            pred_counter.update(preds.cpu().tolist())
            label_counter.update(labels.cpu().tolist())

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            n_steps += 1
            
    if(verbose):
        print(f"Val pred: [{pred_counter[0]}, {pred_counter[1]}] vs. labels: [{label_counter[0]}, {label_counter[1]}]")

    return (
        total_loss / max(n_steps, 1),
        correct / max(total, 1),
    )

def compute_metrics(true_labels, predictions, probabilities):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)

    try:
        roc_auc = roc_auc_score(true_labels, probabilities)
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp

def split_feature_datasets(mapped_csv_path, train_fraction=0.85, seed=42):
    train_dataset = CamelyonPatchFeatureDataset(mapped_csv_path, split="train")
    val_dataset = CamelyonPatchFeatureDataset(mapped_csv_path, split="val")
    test_dataset = CamelyonPatchFeatureDataset(mapped_csv_path, split="test")

    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        raise ValueError(f"Empty split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    print(f"Feature datasets | train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def x_split_feature_datasets(mapped_csv_path, train_fraction=0.85, seed=42):
    base_train = CamelyonPatchFeatureDataset(mapped_csv_path, split="train")
    test_dataset = CamelyonPatchFeatureDataset(mapped_csv_path, split="test")

    train_size = int(train_fraction * len(base_train) + 0.5)
    val_size = len(base_train) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Train/validation split is too small: train={train_size}, val={val_size}."
        )
    if len(test_dataset) == 0:
        raise ValueError("No test_* slides were found in the mapped CSV.")

    train_dataset, val_dataset = torch.utils.data.random_split(
        base_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_dataset, val_dataset, test_dataset


# to have the weighted loss for validation loss like train loss
def weighted_cross_entropy_eval(logits, labels, pos_weight=1.0):
    log_probs = F.log_softmax(logits, dim=1)

    nll = -log_probs.gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)

    weights = (
        1.0
        + labels.to(logits.dtype)
        * (pos_weight - 1.0)
    )

    return (nll * weights).sum() / weights.sum()

def print_bag_stats(name, dataset):
    labels = [
        int(p["label"])
        for p in dataset.bag_plan
    ]

    slides = [
        p["slide_id"]
        for p in dataset.bag_plan
    ]

    print(f"\n{name}")
    print("-" * 50)
    print("Total bags :", len(dataset.bag_plan))
    print("Bag labels :", Counter(labels))

    df = pd.DataFrame(dataset.bag_plan)

    print("\nBags per slide:")
    print(
        df.groupby(["label", "slide_id"])
          .size()
          .groupby(level=0)
          .describe()
    )
    
def aggregate_slide_predictions(
    slide_ids,labels,probabilities,threshold=0.5,method="max",top_k=3,
):
    slide_records = defaultdict(
        lambda: {
            "labels": [],
            "probabilities": [],
        }
    )

    for slide_id, label, prob in zip(
        slide_ids, labels, probabilities,
    ):
        slide_id = str(slide_id)

        slide_records[slide_id]["labels"].append(int(label))
        slide_records[slide_id]["probabilities"].append(float(prob))

    slide_ids_out = []
    slide_labels = []
    slide_predictions = []
    slide_probabilities = []

    for slide_id, record in slide_records.items():
        unique_labels = set(record["labels"])
        if len(unique_labels) != 1:
            raise ValueError(
                f"{slide_id}: inconsistent labels: {unique_labels}"
            )
        slide_label = record["labels"][0]
        probs_this_slide = np.asarray(
            record["probabilities"],
            dtype=np.float32,
        )

        if method == "max":
            slide_prob = float(probs_this_slide.max())
        elif method == "mean":
            slide_prob = float(probs_this_slide.mean())
        elif method == "topk_mean":
            k = min(top_k, len(probs_this_slide))
            top_probs = np.sort(probs_this_slide)[-k:]
            slide_prob = float(top_probs.mean())
        else:
            raise ValueError(
                f"Unknown aggregation method: {method}"
            )

        slide_pred = int(slide_prob >= threshold)

        slide_ids_out.append(slide_id)
        slide_labels.append(slide_label)
        slide_predictions.append(slide_pred)
        slide_probabilities.append(slide_prob)

    return (
        slide_ids_out,
        slide_labels,
        slide_predictions,
        slide_probabilities,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    expr_type = "virchow_features_came16_17"
    os.makedirs(expr_type, exist_ok=True)

    MODE = "train"  # "train" on IPU box; "compile" on RBLN NPU box
    model_name = "feature_transformer"

    # ------------------------------------------------------------------
    # Feature-data paths and model shape
    # ------------------------------------------------------------------
    mapped_csv_out = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/annotations/combined/came16_came17_all.csv"
    
    # One slide -> one fixed bag. Start conservatively; increase later if useful.
    BAG_SIZE = 256
    
    TRAIN_BAGS_PER_SLIDE = 8
    VAL_BAGS_PER_SLIDE = 4
    TEST_BAGS_PER_SLIDE = 4

    # ResNet50 penultimate features are normally 2048-D, but the script infers
    # the actual dimension from your saved tensors and checks it.
    EMBED_DIM = 768
    NUM_HEADS = 12
    NUM_LAYERS = 12
    MLP_RATIO = 4
    NUM_IPUS = 4
    DROPOUT = 0.4
    USE_COORDS = True

    # ------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------
    num_epochs = 50
    current_decay = 1e-3

    rep_fact = 1
    device_itr = 4
    grad_accum = 32

    # Feature bags are far smaller than raw 224x224 images, so batch sizes >1
    # are realistic. Start small and scale after the first successful compile.
    loop_batch_sizes = [1, 2]
    loop_learn_rates = [5e-5, 1e-5, 5e-6]
    learn_rate = 5e-5

    patience = 13
    lr_patience = 4
    lr_factor = 0.5
    min_lr = 1e-7

    # No. of negative / No. of positive patches. keep is as 1.0 and balance the class with neg_bags_per_slide
    pos_weight = 1.0  
    
    #each negative slide should contribute 6 different negative bags instead of just 1.
    neg_bags_per_slide = 6
    loop_neg_bags_per_slide = [6]
    
    # ------------------------------------------------------------------
    # Dataset is needed in both modes so FEATURE_DIM and split sizes match.
    # ------------------------------------------------------------------
    train_dataset, val_dataset, test_dataset = split_feature_datasets(mapped_csv_out)

    FEATURE_DIM = infer_feature_dim(train_dataset)
    dataset_type = os.path.splitext(os.path.basename(mapped_csv_out))[0]

    print(f"Mapped CSV       : {mapped_csv_out}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Bag size         : {BAG_SIZE}")
    print(
        f"Slides           : train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    print("Train labels     :", Counter(get_labels(train_dataset)))
    print("Val labels       :", Counter(get_labels(val_dataset)))
    print("Test labels      :", Counter(get_labels(test_dataset)))

    train_data = FixedFeatureBagDataset(
            train_dataset, bag_size=BAG_SIZE, training=True, seed=42,
            positive_csv="./positive_feature_audit/positive_feature_indices.csv", pos_ratio=0.10, 
            patch_spacing=256, cluster_gap_patches=2, context_radius_patches=3, 
            min_bags_per_cluster=1, max_bags_per_cluster=8, max_bags_per_slide=16,
        )
    
    val_data = FixedFeatureBagDataset(
            val_dataset, bag_size=BAG_SIZE, training=False, seed=42,
            positive_csv="./positive_feature_audit/positive_feature_indices.csv", pos_ratio=0.10, 
            patch_spacing=256, cluster_gap_patches=2, context_radius_patches=3,
            min_bags_per_cluster=1, max_bags_per_cluster=8, max_bags_per_slide=16,
        )

    test_data = FixedFeatureBagDataset(
            test_dataset, bag_size=BAG_SIZE, training=False, seed=42,
            positive_csv="./positive_feature_audit/positive_feature_indices.csv", pos_ratio=0.10, 
            patch_spacing=256, cluster_gap_patches=2, context_radius_patches=3,
            min_bags_per_cluster=1, max_bags_per_cluster=8, max_bags_per_slide=16,
        )
    
    plan_df = pd.DataFrame(train_data.bag_plan)
    print(plan_df.groupby(["label", "slide_id"]).size().describe())
    
    print(f"\n⛷ Total train bags: {len(train_data)}")
    print_bag_stats("TRAIN", train_data)
    print_bag_stats("VAL", val_data)
    print_bag_stats("TEST", test_data)

    # ==================================================================
    # IPU TRAINING
    # ==================================================================
    if MODE == "train":
        if poptorch is None:
            raise ImportError("poptorch is required for MODE='train'.")

        print("☕  PopTorch:", poptorch.__file__)
        print("Version :", getattr(poptorch, "__version__", "unknown"))

        for batch_size in loop_batch_sizes:
            accum_mapping = {2:16, 4:8, 8:4}
            grad_accum = accum_mapping.get(batch_size, 8)
                
            for learn_rate in loop_learn_rates:
                print("\n" + "-" * 100)
                print(
                    f"🚩🚩   Feature Transformer | batch={batch_size} | lr={learn_rate} | "
                    f"neg bags per slides={neg_bags_per_slide} | "
                    f"bag={BAG_SIZE} | feat_dim={FEATURE_DIM} | rf={rep_fact} | "
                    f"di={device_itr} | ga={grad_accum}"
                )
                print("-" * 100)

                # Four IPUs are required because the feature Transformer is
                # explicitly pipelined across IPU 0..3.
                train_opts = poptorch.Options()
                train_opts.setExecutionStrategy(poptorch.PipelinedExecution())
                train_opts.replicationFactor(rep_fact)
                train_opts.deviceIterations(device_itr)
                train_opts.Training.gradientAccumulation(grad_accum)
                train_opts.Precision.enableStochasticRounding(True)
                train_opts.Precision.setPartialsType(torch.half)
                train_opts.setAvailableMemoryProportion({f"IPU{i}": 0.2 for i in range(4)})
                train_opts.enableExecutableCaching("./poptorch_cache_feature")

                valid_opts = poptorch.Options()
                valid_opts.setExecutionStrategy(poptorch.PipelinedExecution())
                valid_opts.replicationFactor(rep_fact)
                valid_opts.deviceIterations(device_itr)
                valid_opts.Precision.setPartialsType(torch.half)
                valid_opts.setAvailableMemoryProportion({f"IPU{i}": 0.2 for i in range(4)})
                valid_opts.enableExecutableCaching("./poptorch_cache_feature")

                
                # class_weights, feature_dim=1024, embed_dim=768, num_heads=12,
                # num_layers=12, mlp_ratio=4, num_classes=2, dropout=0.0,
                # bag_size=256, num_ipus=4, use_coords=True):

                model_kwargs = dict(
                    feature_dim=FEATURE_DIM, bag_size=BAG_SIZE, embed_dim=EMBED_DIM,
                    num_heads=NUM_HEADS, num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO,
                    dropout=DROPOUT, num_classes=2, use_coords=USE_COORDS, num_ipus=NUM_IPUS
                )
                class_counts = torch.tensor([2490, 946], dtype=torch.float32)
                class_weights = class_counts.sum() / (2.0 * class_counts)

                print(class_weights)

                model = FeatureTransformerCamelyon(class_weights=class_weights, **model_kwargs).half()
                val_model = FeatureTransformerCamelyon(class_weights=class_weights, **model_kwargs).half()
                test_model = FeatureTransformerCamelyon(class_weights=class_weights, **model_kwargs).half()
                model.train()

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=learn_rate, weight_decay=current_decay
                )
                poptorch_model = poptorch.trainingModel(
                    model, options=train_opts, optimizer=optimizer
                )

                train_loader = poptorch.DataLoader(
                    train_opts, train_data, batch_size=batch_size, shuffle=True,
                    drop_last=True, mode=poptorch.DataLoaderMode.Async,
                    async_options={"buffer_size": 2},
                )
                
                val_loader = poptorch.DataLoader(
                    valid_opts, val_data, batch_size=batch_size, shuffle=False,
                    drop_last=True, mode=poptorch.DataLoaderMode.Async,
                )
                
                test_loader = poptorch.DataLoader(
                    valid_opts, test_data, batch_size=batch_size, shuffle=False,
                    drop_last=True, mode=poptorch.DataLoaderMode.Async,
                )

                PT_MODEL_PATH = (
                    f"./{expr_type}/{model_name}_fd{FEATURE_DIM}_bag{BAG_SIZE}_"
                    f"bs{batch_size}_lr{learn_rate}_rf{rep_fact}_di{device_itr}_"
                    f"ga{grad_accum}_{dataset_type}.pt"
                )

                val_model.load_state_dict(model.state_dict())
                val_model.eval()
                poptorch_val = poptorch.inferenceModel(val_model, options=valid_opts)
                val_has_compiled = False

                best_val_loss = float("inf")
                epochs_without_improvement = 0
                epochs_without_lr_improvement = 0
                current_lr = learn_rate
                start_time = time.time()
                
                print("Training slides :", len(train_dataset))
                print("Training bags   :", len(train_data))
                print("Bags/slide      :", TRAIN_BAGS_PER_SLIDE)
                print("Features/bag    :", BAG_SIZE)

                print("Validation slides:", len(val_dataset))
                print("Validation bags  :", len(val_data))
                
                for epoch in range(num_epochs):
                    train_data.set_epoch(epoch)
                    running_loss = 0.0
                    n_steps = 0
                    time0 = time.time()
                    poptorch_model.train()

                    for features, coords, mask, labels, slide_indices in train_loader:
                        features = features.half()
                        coords = coords.half()
                        mask = mask.half()
                        _, loss = poptorch_model(features, coords, mask, labels)
                        running_loss += loss.mean().item()
                        n_steps += 1

                    train_loss = running_loss / max(n_steps, 1)

                    # Synchronize training weights into a separate validation model.
                    poptorch_model.copyWeightsToHost()
                    val_model.load_state_dict(model.state_dict())
                    poptorch_model.detachFromDevice()

                    if val_has_compiled:
                        poptorch_val.attachToDevice()
                        poptorch_val.copyWeightsToDevice()

                    val_loss, val_acc = evaluate_ipu(poptorch_val, val_loader, verbose=True)
                  
                    val_has_compiled = True

                    poptorch_val.detachFromDevice()
                    poptorch_model.attachToDevice()

                    epoch_time = time.time() - time0
                    print(
                        f"Epoch {epoch+1:03d}/{num_epochs} | "
                        f"train_loss={train_loss:.4f} | "
                        f"val_loss={val_loss:.4f} | " 
                        f"val_acc={val_acc:.4f} | "                      
                        f"lr={current_lr:.2e} | "
                        f"elapsed={epoch_time:.1f}s"
                    )
                
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        epochs_without_lr_improvement = 0
                        poptorch_model.copyWeightsToHost()
                        torch.save(model.state_dict(), PT_MODEL_PATH)
                        print(f"✅  Saved best model: {PT_MODEL_PATH}")
                    else:
                        epochs_without_improvement += 1
                        epochs_without_lr_improvement += 1

                    if epochs_without_lr_improvement >= lr_patience:
                        new_lr = max(current_lr * lr_factor, min_lr)
                        if new_lr < current_lr:
                            current_lr = new_lr
                            for group in optimizer.param_groups:
                                group["lr"] = current_lr
                            poptorch_model.setOptimizer(optimizer)
                            epochs_without_lr_improvement = 0
                            print(f"🔽 Reducing LR to {current_lr:.2e}")

                    if epochs_without_improvement >= patience:
                        print("Early stopping triggered.")
                        break

                print(f"Total train time: {(time.time() - start_time) / 60.0:.1f} min")

                # Release training/validation executables before final test.
                cleanup_poptorch(
                    ("poptorch_model", poptorch_model),
                    ("poptorch_val", poptorch_val),
                )

                print(f"🌱 Loading best checkpoint for IPU test: {PT_MODEL_PATH}")
                state = torch.load(PT_MODEL_PATH, map_location="cpu")
                test_model.load_state_dict(state)
                test_model.eval()
                infer_model = poptorch.inferenceModel(test_model, options=valid_opts)

                predictions = []
                probabilities = []
                true_labels = []
                test_start = time.time()
                test_slide_ids = []

                with torch.no_grad():
                    for features, coords, mask, labels, slide_indices in test_loader:
                        logits = infer_model(
                            features.half(), coords.half(), mask.half()
                        ).float()
                        probs = torch.softmax(logits, dim=1)
                        preds = torch.argmax(logits, dim=1)
                        predictions.extend(preds.cpu().tolist())
                        probabilities.extend(probs[:, 1].cpu().tolist())
                        true_labels.extend(labels.cpu().tolist())
                        for slide_idx in slide_indices.cpu().tolist():
                            test_slide_ids.append(
                                test_data.idx_to_slide_id[int(slide_idx)]
                            )

                test_elapsed = time.time() - test_start
                (
                    accuracy,
                    precision,
                    recall,
                    f1,
                    roc_auc,
                    tn,
                    fp,
                    fn,
                    tp,
                ) = compute_metrics(true_labels, predictions, probabilities)
                              
                print("-" * 60)
                print(f"accuracy  : {accuracy:.4f}")
                print(f"precision  {precision:.4f}")
                print(f"recall    : {recall:.4f}")
                print(f"f1        : {f1:.4f}")
                print(f"roc_auc   : {roc_auc:.4f}")
                print(f"Confusion matrix   : \n [{tn}, {fp}] \n [{fn}, {tp}]")
                print(
                    f"IPU throughput     : {len(true_labels) / test_elapsed:.2f} slides/sec"
                )
                print("-" * 60)

                # SLIDE-level aggregation
                (
                    slide_ids_out,
                    slide_labels,
                    slide_predictions,
                    slide_probabilities,
                ) = aggregate_slide_predictions(
                    slide_ids=test_slide_ids,
                    labels=true_labels,
                    probabilities=probabilities,
                    threshold=0.5,
                    method="max",
                )
                
                # SLIDE-level metrics
                (
                    slide_accuracy,
                    slide_precision,
                    slide_recall,
                    slide_f1,
                    slide_roc_auc,
                    slide_tn,
                    slide_fp,
                    slide_fn,
                    slide_tp,
                ) = compute_metrics(
                    slide_labels,
                    slide_predictions,
                    slide_probabilities,
                )
                
                print("-" * 60)
                print(f"Slide accuracy  : {slide_accuracy:.4f}")
                print(f"Slide precision  {slide_precision:.4f}")
                print(f"Slide recall    : {slide_recall:.4f}")
                print(f"Slide f1        : {slide_f1:.4f}")
                print(f"Slide roc_auc   : {slide_roc_auc:.4f}")
                print(f"Slide confusion matrix   : \n [{slide_tn}, {slide_fp}] \n [{slide_fn}, {slide_tp}]")
                print("-" * 60)
                
                
                CSV_FILE = f"./{expr_type}/ipu_vs_npu_feature_benchmark.csv"
                pd.DataFrame(
                    [
                        {
                            "device": "IPU",
                            "model": model_name,
                            "dataset": dataset_type,
                            "feature_dim": FEATURE_DIM,
                            "bag_size": BAG_SIZE,
                            "batch_size": batch_size,
                            "learn_rate": learn_rate,
                            "neg_bags_per_slide":neg_bags_per_slide,
                            "rep_fact": rep_fact,
                            "device_itr": device_itr,
                            "grad_accum": grad_accum,
                            "accuracy": f'{accuracy:.4f}',
                            "precision": f'{precision:.4f}',
                            "recall": f'{recall:.4f}',
                            "f1": f'{f1:.4f}',
                            "roc_auc": f'{roc_auc:.4f}',
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "tp": tp,
                            "Slide accuracy"  : f"{slide_accuracy:.4f}",
                            "Slide precision" : f"{slide_precision:.4f}",
                            "Slide recall"    : f"{slide_recall:.4f}",
                            "Slide f1"        : f"{slide_f1:.4f}",
                            "Slide roc_auc"   : f"{slide_roc_auc:.4f}",
                            "Slide tn": slide_tn,
                            "Slide fp": slide_fp,
                            "Slide fn": slide_fn,
                            "Slide tp": slide_tp,                         
                            "slides_per_sec": f'{(len(true_labels) / test_elapsed):.2f}',                         
                            "checkpoint": PT_MODEL_PATH,                          
                        }
                    ]
                ).to_csv(
                    CSV_FILE,
                    mode="a",
                    header=not os.path.exists(CSV_FILE),
                    index=False,
                )
                
                cleanup_poptorch(("infer_model", infer_model))

        print(
            f"🏄‍♀️ Done. Copy the .pt checkpoint and mapped feature data to the NPU box, "
            f"then set MODE='compile'."
        )

    # ==================================================================
    # RBLN NPU COMPILE + TEST
    # ==================================================================
    elif MODE == "compile":
        import rebel

        # These values MUST match the IPU experiment/checkpoint you copied.
        batch_size = 1
        learn_rate = 1e-4

        PT_MODEL_PATH = (
            f"./{expr_type}/{model_name}_fd{FEATURE_DIM}_bag{BAG_SIZE}_"
            f"bs{batch_size}_lr{learn_rate}_rf{rep_fact}_di{device_itr}_"
            f"ga{grad_accum}_{dataset_type}.pt"
        )
        if not os.path.exists(PT_MODEL_PATH):
            raise FileNotFoundError(
                f"Can't find {PT_MODEL_PATH}. Copy the IPU checkpoint to this box."
            )

        RBLN_MODEL_PATH = PT_MODEL_PATH.replace(".pt", ".rbln")

        model = FeatureTransformerCamelyon(
            feature_dim=FEATURE_DIM,
            bag_size=BAG_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            mlp_ratio=MLP_RATIO,
            dropout=DROPOUT,
            num_classes=2,
            use_coords=USE_COORDS,
            ipu_pipeline=False,
        )
        print(f"Loading checkpoint: {PT_MODEL_PATH}")
        model.load_state_dict(torch.load(PT_MODEL_PATH, map_location="cpu"))
        model.eval()

        if os.path.exists(RBLN_MODEL_PATH):
            print(f"Loading existing compiled model: {RBLN_MODEL_PATH}")
            runtime = rebel.Runtime(RBLN_MODEL_PATH)
        else:
            print("Compiling feature Transformer with RBLN SDK...")
            input_info = [
                ("features", [1, BAG_SIZE, FEATURE_DIM], "float32"),
                ("coords", [1, BAG_SIZE, 2], "float32"),
                ("mask", [1, BAG_SIZE], "float32"),
            ]
            compiled = rebel.compile_from_torch(
                model,
                input_info,
                model_trace_method="jittrace",
            )
            compiled.save(RBLN_MODEL_PATH)
            runtime = rebel.Runtime(RBLN_MODEL_PATH)

        predictions = []
        probabilities = []
        true_labels = []
        start_time = time.time()

        # RBLN was compiled for batch=1 here: one slide-feature bag at a time.
        for idx in range(len(test_data)):
            features, coords, mask, label = test_data[idx]
            x_features = features.unsqueeze(0).numpy().astype(np.float32)
            x_coords = coords.unsqueeze(0).numpy().astype(np.float32)
            x_mask = mask.unsqueeze(0).numpy().astype(np.float32)

            outputs = runtime(x_features, x_coords, x_mask)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            outputs = np.asarray(outputs)
            if outputs.ndim == 1:
                outputs = outputs.reshape(1, -1)

            # Stable 2-class softmax on CPU for metrics.
            shifted = outputs - outputs.max(axis=1, keepdims=True)
            exp_outputs = np.exp(shifted)
            probs = exp_outputs / exp_outputs.sum(axis=1, keepdims=True)

            predictions.append(int(outputs.argmax(axis=1)[0]))
            probabilities.append(float(probs[0, 1]))
            true_labels.append(int(label.item()))

        elapsed = time.time() - start_time
        (
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            tn,
            fp,
            fn,
            tp,
        ) = compute_metrics(true_labels, predictions, probabilities)

        print("-" * 60)
        print(f"NPU slide accuracy : {accuracy:.4f}")
        print(f"NPU precision      : {precision:.4f}")
        print(f"NPU recall         : {recall:.4f}")
        print(f"NPU f1             : {f1:.4f}")
        print(f"NPU roc_auc        : {roc_auc:.4f}")
        print(f"Confusion matrix   : tn={tn}, fp={fp}, fn={fn}, tp={tp}")
        print(f"NPU throughput     : {len(test_data) / elapsed:.2f} slides/sec")
        print("-" * 60)

        CSV_FILE = f"./{expr_type}/ipu_vs_npu_feature_benchmark.csv"
        pd.DataFrame(
            [
                {
                    "device": "NPU",
                    "model": model_name,
                    "dataset": dataset_type,
                    "feature_dim": FEATURE_DIM,
                    "bag_size": BAG_SIZE,
                    "batch_size": 1,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "slides_per_sec": len(test_data) / elapsed,
                    "checkpoint": PT_MODEL_PATH,
                    "rbln_model": RBLN_MODEL_PATH,
                }
            ]
        ).to_csv(
            CSV_FILE,
            mode="a",
            header=not os.path.exists(CSV_FILE),
            index=False,
        )

    else:
        raise ValueError(f"Unknown MODE: {MODE!r}. Use 'train' or 'compile'.")


if __name__ == "__main__":
    main()

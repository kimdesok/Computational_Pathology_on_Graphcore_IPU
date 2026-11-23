import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.spatial import cKDTree

import sys
sys.path.append("/opt/pytorch/lib/python3.8/site-packages")

import poptorch
from torch_geometric.nn import GCNConv, global_mean_pool

from pathlib import Path
import pandas as pd
from scipy.spatial import cKDTree

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import time

# --- Hyperparameters & Static IPU Shapes ---
K_NEIGHBORS = 4
MAX_PATCHES = 5000   # Fixed number of patches per WSI bag

FEATURE_DIM = 1024   # e.g., 1024 for ResNet50/UNI/CONCH
BATCH_SIZE = 2
LR = 5e-5
EPOCHS = 200
POS_WEIGHT = 1.375

patience = 29
lr_patience = 4
lr_factor, min_lr = 0.5, 1e-08

rept_fac=1
device_itr=4
grad_accum=16

loop_batch_sizes = [4]
loop_pos_weights = [1.3, 1.4, 1.5, 1.6, 1.7]
loop_learn_rates = [5e-03, 1e-3, 5e-4, 1e-4]

#initialization
#pos_weight = 1.0
#batch_size = 1
#learn_rate = 1.0e-4

# directories
SAVE_MODEL_DIR = "/home/work/mytest/gnn/models"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
SAVE_RESULT_DIR = "/home/work/mytest/gnn/logs"
os.makedirs(SAVE_RESULT_DIR, exist_ok=True)

# ==============================================================================
# 1. LOCAL H5 WSI DATASET (READS FROM RAW, CACHES TO PROCESSED)
# ==============================================================================

import torch
from torch.utils.data import Dataset
import pandas as pd

class CamelyonPatchFeatureDataset(Dataset):
    """
    PyTorch Dataset driven strictly by the mapped CSV file.
    
    Parameters:
    -----------
    mapped_csv_path : str
        Path to the 'came16_labels_mapped.csv' generated in Step 1.
    split : str
        'train'  -> Loads tumor_* and normal_* slides.
        'test'   -> Loads test_* slides for inference.
        'all'    -> Loads all available slides in the mapped CSV.
    """
    def __init__(self, mapped_csv_path: str, split: str = 'train'):
        super().__init__()
        self.df = pd.read_csv(mapped_csv_path)
        self.split = split.lower()
        
        required_columns = {
            "file_path",
            "coord_path",
            "slide_id",
            "label"
        }
        
        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Mapped CSV is missing columns: {missing}. "
                f"Rebuild {mapped_csv_path}."
            )
        
        # Filter dataframe by split
        if self.split == 'train':
            self.df = self.df[~self.df['slide_id'].str.startswith('test_')].reset_index(drop=True)
        elif self.split == 'test':
            self.df = self.df[self.df['slide_id'].str.startswith('test_')].reset_index(drop=True)
        elif self.split == 'all':
            pass
        else:
            raise ValueError(f"Invalid split '{split}'. Choose from ['train', 'test', 'all'].")

        #print(f"[{self.split.upper()} Split] Total slides: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        row = self.df.iloc[idx]

        file_path = row["file_path"]
        coord_path = row["coord_path"]

        label = int(row["label"])
        slide_id = row["slide_id"]

        payload = torch.load(
            file_path,
            map_location="cpu"
        )

        if isinstance(payload, dict):
            features = payload["features"]
        else:
            features = payload

        with h5py.File(coord_path, "r") as f:
            coords = torch.from_numpy(
                f["coords"][:]
            ).long()

        assert features.shape[0] == coords.shape[0], (
            f"{slide_id}: "
            f"{features.shape[0]} features != "
            f"{coords.shape[0]} coords"
        )

        return {
            "features": features,
            "coords": coords,
            "label": torch.tensor(label, dtype=torch.long),
            "slide_id": slide_id,
            "center": row.get("center", -1),
        }
    

def build_camelyon16_mapped_csv(
    annotation_csv_path: str,
    pt_dir: str,
    coord_dir: str,
    output_csv_path: str
) -> pd.DataFrame:
    """
    Merges annotation metadata CSV with local .pt feature paths by matching slide IDs.
    
    Expected CSV Columns: ['image', 'type', 'class', 'center']
    Example values: 'tumor_001.tif', 'tumor', 'micro', 1
    """
    # 1. Load annotation CSV (Support comma-separated or tab-separated)
    sep = "\t" if annotation_csv_path.endswith(".tsv") else ","
    df_meta = pd.read_csv(annotation_csv_path, sep=sep)
    
    # Clean string columns
    df_meta['image'] = df_meta['image'].astype(str).str.strip()
    df_meta['type'] = df_meta['type'].astype(str).str.strip().str.lower()
    
    # Extract core slide_id without extensions (e.g. 'tumor_001.tif' -> 'tumor_001')
    df_meta['slide_id'] = df_meta['image'].apply(lambda x: Path(x).stem)
    
    # 2. Map Binary Labels (tumor/macro/micro -> 1, normal/negative -> 0)
    def map_label(row):
        t_val = str(row['type']).lower()
        c_val = str(row.get('class', '')).lower()
        if t_val in ['tumor', 'macro', 'micro'] or c_val in ['macro', 'micro', 'tumor']:
            return 1
        elif t_val in ['normal', 'negative'] or c_val == 'negative':
            return 0
        else:
            raise ValueError(f"Unknown label category: type={t_val}, class={c_val}")

    df_meta['label'] = df_meta.apply(map_label, axis=1)

    # 3. Gather local .pt feature files
    pt_files = list(Path(pt_dir).glob("*.pt"))

    if not pt_files:
        raise FileNotFoundError(
            f"No .pt files found in directory: {pt_dir}"
        )

    records = []

    for p in pt_files:

        slide_id = p.stem
        coord_path = Path(coord_dir) / f"{slide_id}.h5"

        if not coord_path.exists():
            print(
                f"WARNING: coordinate file missing for {slide_id}"
            )
            continue

        records.append({
            "file_path": str(p.resolve()),
            "coord_path": str(coord_path.resolve()),
            "slide_id": slide_id
        })

    df_local = pd.DataFrame(records)

    #print(
    #    f"Matched {len(df_local)} "
    #    f"feature/coordinate pairs."
    #)

    # 4. Merge metadata with local file paths on slide_id
    df_dataset = pd.merge(df_local, df_meta[['slide_id', 'label', 'type', 'class', 'center']], on="slide_id", how="inner")

    # 5. Save the linked index
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_dataset.to_csv(output_csv_path, index=False)
    #print(f"Successfully mapped {len(df_dataset)} local .pt files out of {len(pt_files)} found.")
    
    return df_dataset


def build_fixed_knn_edges(
    coords,
    num_real_nodes,
    max_nodes=MAX_PATCHES,
    k=K_NEIGHBORS
):

    coords = coords[:num_real_nodes].cpu().numpy()
    tree = cKDTree(coords)
    k_query = min(k + 1, num_real_nodes)
    _, neighbors = tree.query(
        coords,
        k=k_query
    )

    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]

    if k_query > 1:
        neighbors = neighbors[:, 1:]
    else:
        neighbors = np.empty( (num_real_nodes, 0), dtype=np.int64)

    if neighbors.shape[1] < k:

        missing = k - neighbors.shape[1]

        self_idx = np.arange(num_real_nodes)[:, None]

        padding = np.repeat(self_idx,missing, axis=1)

        neighbors = np.concatenate(
            [neighbors, padding],
            axis=1
        )

    neighbors = neighbors[:, :k]

    src_real = np.repeat(
        np.arange(num_real_nodes),
        k
    )

    dst_real = neighbors.reshape(-1)

    if num_real_nodes < max_nodes:

        padded = np.arange(
            num_real_nodes,
            max_nodes
        )

        src_pad = np.repeat(
            padded,
            k
        )

        dst_pad = src_pad.copy()

        src = np.concatenate(
            [src_real, src_pad]
        )

        dst = np.concatenate(
            [dst_real, dst_pad]
        )

    else:
        src = src_real
        dst = dst_real

    return torch.tensor(
        np.stack([src, dst]),
        dtype=torch.long
    )

def ipu_collate_fn(batch):

    padded_features = []
    batch_edges = []
    labels = []

    for batch_idx, sample in enumerate(batch):
        feat = sample["features"]
        coords = sample.get("coords", None)
        label = sample["label"]
        if coords is None:
            raise ValueError(
                "GNN requires patch coordinates, "
                "but coords were not found in the .pt file."
            )
        num_patches = feat.size(0)
        
        # =====================================================
        # RANDOM NODE/PATCH SAMPLING
        # =====================================================
        if num_patches > MAX_PATCHES:
            idx = torch.randperm(num_patches)[:MAX_PATCHES]
            # use exactly the same indices for features and coordinates
            feat = feat[idx]
            coords = coords[idx]
            num_real = MAX_PATCHES
        else:
            num_real = num_patches

        # --------------------------------------------
        # Keep current sampling behavior for now.
        #
        # Important:
        # Do NOT introduce random sampling yet.
        # We want the GNN vs baseline comparison
        # to change only the architecture.
        # --------------------------------------------
        num_real = min(
            num_patches,
            MAX_PATCHES
        )

        feat = feat[:num_real]
        coords = coords[:num_real]
        
        if not hasattr(ipu_collate_fn, "_printed_coords"):
            print("\n--- Coordinate debug ---")
            print("Slide:", sample.get("slide_id", "unknown"))
            print("num_real:", num_real)
            print("coords.shape:", coords.shape)
            #print("First 10 coordinates:")
            #print(coords[:10])            
            ipu_collate_fn._printed_coords = True

        # --------------------------------------------
        # Pad feature nodes
        # --------------------------------------------
        if num_real < MAX_PATCHES:
            pad_size = MAX_PATCHES - num_real
            feat = F.pad(
                feat,
                (0, 0, 0, pad_size)
            )
        padded_features.append(feat)

        # --------------------------------------------
        # Build spatial graph
        # --------------------------------------------
        edge_index = build_fixed_knn_edges(
            coords=coords,
            num_real_nodes=num_real,
            max_nodes=MAX_PATCHES,
            k=K_NEIGHBORS
        )

        if not hasattr(ipu_collate_fn, "_printed_edges"):
            print("\n--- Edge debug ---")
            print("edge_index.shape:", edge_index.shape)
            print("First 20 source nodes:")
            print(edge_index[0, :20])
            print("First 20 destination nodes:")
            print(edge_index[1, :20])
            ipu_collate_fn._printed_edges = True

        batch_edges.append(edge_index)
        labels.append(label)

    # [B, N, F]
    bx = torch.stack(padded_features,dim=0)

    # [2, B * N * K]
    bedge = torch.stack(batch_edges,dim=0)

    # [B]
    by = torch.stack(labels)
    
    # -------------------------------------------
    # DEBUG: print only first collated batch
    # -------------------------------------------
    if not hasattr(ipu_collate_fn, "_printed_debug"):
        print("\n--- GNN batch debug ---")
        print("bx.shape    :", bx.shape)
        print("bedge.shape :", bedge.shape)
        print("by.shape    :", by.shape)
        ipu_collate_fn._printed_debug = True

    return bx, bedge, by

def ipu_collate_fn_train(batch):

    padded_features = []
    batch_edges = []
    labels = []

    for batch_idx, sample in enumerate(batch):

        feat = sample["features"]
        coords = sample["coords"]
        label = sample["label"]

        num_patches = feat.size(0)

        # =====================================================
        # TRAINING:
        # Randomly sample MAX_PATCHES from the slide
        # =====================================================
        if num_patches > MAX_PATCHES:
            idx = torch.randperm(num_patches)[:MAX_PATCHES]
            # Very important:
            # feature and coordinate use SAME indices
            feat = feat[idx]
            coords = coords[idx]
            num_real = MAX_PATCHES
        else:
            num_real = num_patches

        # =====================================================
        # Pad slides containing fewer than MAX_PATCHES
        # =====================================================
        if num_real < MAX_PATCHES:

            pad_size = MAX_PATCHES - num_real

            feat = F.pad(
                feat,
                (0, 0, 0, pad_size)
            )

        padded_features.append(feat)

        # =====================================================
        # Construct spatial k-NN graph
        # =====================================================
        edge_index = build_fixed_knn_edges(
            coords=coords,
            num_real_nodes=num_real,
            max_nodes=MAX_PATCHES,
            k=K_NEIGHBORS
        )

        batch_edges.append(edge_index)
        labels.append(label)

    # =========================================================
    # Build PopTorch batch
    # =========================================================
    bx = torch.stack(
        padded_features,
        dim=0
    )

    bedge = torch.stack(
        batch_edges,
        dim=0
    )

    by = torch.stack(labels)

    return bx, bedge, by

def ipu_collate_fn_eval(batch):

    padded_features = []
    batch_edges = []
    labels = []

    for batch_idx, sample in enumerate(batch):

        feat = sample["features"]
        coords = sample["coords"]
        label = sample["label"]

        num_patches = feat.size(0)

        # =====================================================
        # VALIDATION / TEST:
        # deterministic sampling
        # =====================================================
        if num_patches > MAX_PATCHES:

            # Fixed pseudo-random sampling for validation/test.
            # The same slide will always get the same subset.
            slide_id = sample["slide_id"]

            # Stable deterministic seed from slide ID
            seed = sum(ord(c) for c in slide_id)

            generator = torch.Generator()
            generator.manual_seed(seed)

            idx = torch.randperm( num_patches, generator=generator
            )[:MAX_PATCHES]

            # IMPORTANT: same indices for features and coordinates
            feat = feat[idx]
            coords = coords[idx]

            num_real = MAX_PATCHES

        else:
            num_real = num_patches

        # =====================================================
        # Pad smaller slides
        # =====================================================
        if num_real < MAX_PATCHES:
            pad_size = MAX_PATCHES - num_real
            feat = F.pad(feat, (0, 0, 0, pad_size))

        padded_features.append(feat)

        # =====================================================
        # Build exactly the same graph every evaluation
        # =====================================================
        edge_index = build_fixed_knn_edges(
            coords=coords,num_real_nodes=num_real,
            max_nodes=MAX_PATCHES,k=K_NEIGHBORS
        )

        batch_edges.append(edge_index)
        labels.append(label)

    bx = torch.stack(
        padded_features,
        dim=0
    )

    bedge = torch.stack(
        batch_edges,
        dim=0
    )

    by = torch.stack(labels)

    return bx, bedge, by

def show_labels(ds, name):

    labels = []

    for i in range(len(ds)):
        sample = ds[i]
        labels.append(int(sample["label"]))

    values, counts = np.unique(labels, return_counts=True)

    print(f"\n{name}")
    for v, c in zip(values, counts):
        print(f"  label {v}: {c}")
        

class GatedAttentionPooling(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.attn_a = nn.Linear(in_features, hidden_dim)
        self.attn_b = nn.Linear(in_features, hidden_dim)
        self.attn_c = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        a = torch.tanh(self.attn_a(x))
        b = torch.sigmoid(self.attn_b(x))
        attn_logits = self.attn_c(a * b).squeeze(-1)

        attn_logits = torch.where(
            mask > 0, attn_logits, torch.full_like(attn_logits, -1e4)
        )

        weights = F.softmax(attn_logits, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)

        return pooled
    
class WSIPatchMIL(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2):
        super().__init__()
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.attn_a = nn.Linear(hidden_channels, 64)
        self.attn_b = nn.Linear(hidden_channels, 64)
        self.attn_c = nn.Linear(64, 1)
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def name(self):
        return "뭔지 모를 그냥 모델"

    def forward(self, x, edge_index=None):
        mask = (x.abs().sum(dim=-1) > 0)
        h = F.relu(self.fc_in(x))

        a = torch.tanh(self.attn_a(h))
        b = torch.sigmoid(self.attn_b(h))
        attn = self.attn_c(a * b).squeeze(-1)
        attn = attn.masked_fill(mask == 0, -1e4)
        attn = F.softmax(attn, dim=1)

        slide_rep = torch.sum(h * attn.unsqueeze(-1), dim=1)
        return self.classifier(slide_rep)
    
class WSIPatchGNN(nn.Module):
    def __init__(
            self, in_channels: int, hidden_channels: int = 128,
            num_classes: int = 2, attention_dim: int = 64, dropout =0.2
        ):
        super().__init__()

        self.fc_in = nn.Linear(in_channels, hidden_channels)
        
        self.conv1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)   
        self.dropout = nn.Dropout(dropout)

        # gated attention graph pooling
        self.attention_pool = GatedAttentionPooling(
            in_features=hidden_channels,
            hidden_dim=attention_dim
        )
        
        self.classifier = nn.Linear(
            hidden_channels,
            num_classes
        )

    def name(self):
        return "GNN"
    
    def forward(self, x, edge_index):

        """
        x: [B, MAX_PATCHES, FEATURE_DIM]
        edge_index:   [B, 2, E]
        Example:
            x          = [8, 5000, 1024]
            edge_index = [8, 2, 20000]
        """

        batch_size = x.size(0)
        num_nodes = x.size(1)

        #  ResNet50 features
        h = F.relu(self.fc_in(x))                  # [B, N, H]

        # Flatten batch for GCNConv
        h = h.reshape(batch_size * num_nodes, -1)
        
        # Convert each slide's local node IDs into global IDs
        offsets = torch.arange(batch_size, device=edge_index.device).view(batch_size, 1, 1) * num_nodes
        global_edges = edge_index + offsets
        global_edges = global_edges.permute(1, 0, 2).reshape(2, -1)

        # 3. Graph convolution
        h = self.conv1(h, global_edges)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, global_edges)
        h = F.relu(h)
        h = self.dropout(h)

        # Restore slide structure
        h = h.view(batch_size, num_nodes, -1)      # [B, N, H]

        
        # Mask padded nodes
        # Padded feature vectors are all zeros.
        mask = (x.abs().sum(dim=-1) > 0).to(torch.int32)   # [B, N]

        # Gated attention: nodes -> slide representation
        slide_rep = self.attention_pool(h, mask)   # [B, H]

        # Slide classification
        logits = self.classifier(slide_rep)        # [B, 2]

        return logits

# Custom static GCN Layer designed purely for dense matrix calculations on IPUs
class IPUSparseGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, h, edge_index):
        # 1. Project patch features
        wh = self.linear(h) # Shape: [Batch, Num_Patches, Out_Channels]
        
        source_idx = edge_index[:, 0, :] # Shape: [Batch, Num_Edges]
        target_idx = edge_index[:, 1, :] # Shape: [Batch, Num_Edges]
        
        batch_size, num_edges = source_idx.shape
        _, num_patches, channels = wh.shape
        
        # 2. Gather Step (Fully supported by PopTorch via index_select)
        batch_offsets = torch.arange(batch_size, device=h.device).unsqueeze(-1) * num_patches
        flat_source_idx = (source_idx + batch_offsets).view(-1)
        
        # Prepare target indexes matching the structure for scatter accumulation
        # For torch.scatter_add, the target index must match the broadcast shape of edge_messages
        # Shape needed: [Batch * Num_Edges, Channels]
        flat_target_idx = (target_idx + batch_offsets).view(-1, 1).expand(-1, channels)
        
        flat_wh = wh.view(-1, channels)
        edge_messages = torch.index_select(flat_wh, 0, flat_source_idx) # Shape: [Batch * Num_Edges, Channels]
        
        # 3. Scatter Step: SWAPPED OUT index_add_ FOR THE SUPPORTED scatter_add
        # This allocates a clean base matrix and updates it in a parallel-friendly way
        out_flat = torch.zeros_like(flat_wh)
        out_flat = torch.scatter_add(out_flat, 0, flat_target_idx, edge_messages)
        
        # 4. Reshape back to batch dimensions
        out = out_flat.view(batch_size, num_patches, channels)
        return out


class WSIPatchGCNMIL(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2):
        super().__init__()
        
        # Explicit relative IPU stage targets for IDs 4-7
        self.ipu_stage0 = 0
        self.ipu_stage1 = 1
        self.ipu_stage2 = 2
        self.ipu_stage3 = 3
        
        # Entry anchor companion
        self.entry_identity = nn.Identity() 
        self.stage1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU()
        )
        
        # Swapped out for our specialized PopTorch-friendly Sparse GCN Layers
        self.gcn1 = IPUSparseGCNLayer(hidden_channels, hidden_channels)
        self.gcn2 = IPUSparseGCNLayer(hidden_channels, hidden_channels)
        
        self.attn_a = nn.Linear(hidden_channels, 64)
        self.attn_b = nn.Linear(hidden_channels, 64)
        self.attn_c = nn.Linear(64, 1)
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def name(self):
        return "GCN-MIL"

    def forward(self, x, edge_index):
        # x shape: [8, 5000, 1024]
        # edge_index shape: [8, 2, 20000]
        
        # --- STAGE 0: Entry Anchor ---
        poptorch.BeginBlock(self.entry_identity, ipu_id=self.ipu_stage0)
        mask = (x.abs().sum(dim=-1) > 0)
        h = self.stage1(x)

        # --- STAGE 1: First Graph Convolution Pass ---
        poptorch.BeginBlock(self.gcn1, ipu_id=self.ipu_stage1)
        h = F.relu(self.gcn1(h, edge_index))

        # --- STAGE 2: Second Graph Convolution Pass ---
        poptorch.BeginBlock(self.gcn2, ipu_id=self.ipu_stage2)
        h = F.relu(self.gcn2(h, edge_index))

        # --- STAGE 3: Dual-Stream Attention Pooling ---
        poptorch.BeginBlock(self.attn_a, ipu_id=self.ipu_stage3)
        a = torch.tanh(self.attn_a(h))
        b = torch.sigmoid(self.attn_b(h))
        attn = self.attn_c(a * b).squeeze(-1)
        
        attn = attn.masked_fill(mask == 0, -1e4)
        attn = F.softmax(attn, dim=1)

        slide_rep = torch.sum(h * attn.unsqueeze(-1), dim=1)
        return self.classifier(slide_rep)
    

class TrainingWrapper(nn.Module):

    def __init__(
        self,
        model,
        pos_weight=None
    ):
        super().__init__()

        self.model = model
        self.pos_weight = pos_weight

        # IMPORTANT:
        # no built-in class weight here
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none"
        )

    def forward(self, x, edge_index, y=None):
        logits = self.model(x, edge_index)

        if y is not None:

            # Per-sample CE loss
            loss = self.loss_fn(logits, y)

            # class 0 -> weight 1.0 vs. class 1 -> weight pos_weight
            sample_weights = (1.0 + y.float() * (self.pos_weight - 1.0))
            weighted_loss = (loss * sample_weights).mean()
            return (
                logits, poptorch.identity_loss(weighted_loss, reduction="mean")
            )
        
        print("Pos weight in Training Wrapper", self.pos_weight)
        print("Sample weight :", sample_weights)
        return logits
# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================

def evaluate(
    poptorch_eval_model,
    dataloader,
    split_name="Validation",
    verbose=False
):

    poptorch_eval_model.eval()

    total_val_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for bx, bedge, by in dataloader:

            out = poptorch_eval_model(
                bx,
                bedge,
                by
            )

            if isinstance(out, tuple):
                logits, loss = out
                total_val_loss += loss.mean().item()
            else:
                logits = out

            # ----------------------------------------
            # Probabilities
            # ----------------------------------------
            probs = torch.softmax(
                logits,
                dim=-1
            )

            class1_probs = probs[:, 1]

            preds = logits.argmax(
                dim=-1
            )

            # ----------------------------------------
            # Save CPU results
            # ----------------------------------------
            all_labels.extend(
                by.cpu().numpy().tolist()
            )

            all_preds.extend(
                preds.cpu().numpy().tolist()
            )

            all_probs.extend(
                class1_probs.cpu().numpy().tolist()
            )

    # ============================================================
    # Metrics
    # ============================================================

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    # ROC-AUC requires both classes
    if len(set(all_labels)) > 1:
        roc_auc = roc_auc_score(
            all_labels,
            all_probs
        )
    else:
        roc_auc = float("nan")

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    avg_loss = (
        total_val_loss / len(dataloader)
        if len(dataloader) > 0
        else 0.0
    )

    # ============================================================
    # Prediction counts
    # ============================================================

    pred_0 = sum(
        p == 0 for p in all_preds
    )

    pred_1 = sum(
        p == 1 for p in all_preds
    )

    label_0 = sum(
        y == 0 for y in all_labels
    )

    label_1 = sum(
        y == 1 for y in all_labels
    )

    # ============================================================
    # Print results
    # ============================================================

    if(verbose):
        print(
            f"{split_name} predictions: "
            f"0={pred_0}, 1={pred_1}"
        )

        print(
            f"{split_name} labels:      "
            f"0={label_0}, 1={label_1}"
        )

        print(
            f"\n--- {split_name} Metrics ---"
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1-score : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

        print("\nConfusion Matrix:")
        print(cm)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

def print_payload(pt_files):
    for i, f in enumerate(pt_files[:2]):

        #print("\n" + "=" * 70)
        #print(f"FILE {i + 1}: {os.path.basename(f)}")
        #print("=" * 70)

        payload = torch.load(f, map_location="cpu")

        print("Payload type:", type(payload))

        # --------------------------------------------------------
        # Dictionary payload
        # --------------------------------------------------------
        if isinstance(payload, dict):

            print("Keys:", list(payload.keys()))

            for key, value in payload.items():

                print(f"\n  key: {key}")
                print(f"  type: {type(value)}")

                if torch.is_tensor(value):
                    print(f"  shape: {value.shape}")
                    print(f"  dtype: {value.dtype}")

                elif isinstance(value, np.ndarray):
                    print(f"  shape: {value.shape}")
                    print(f"  dtype: {value.dtype}")

                else:
                    print(f"  value: {value}")

        # --------------------------------------------------------
        # Direct tensor payload
        # --------------------------------------------------------
        elif torch.is_tensor(payload):

            print("Direct tensor")
            print("Shape:", payload.shape)
            print("dtype:", payload.dtype)

        # --------------------------------------------------------
        # Other payload
        # --------------------------------------------------------
        else:

            print("Unknown/other payload structure:")
            print(payload)

# ==============================================================================
# 3. PIPELINE INITIALIZATION
# ==============================================================================
pt_directory = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/r50/pt_files"
coord_directory = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/patches"
model_directory = "/home/work/mytest/gnn/models"

mapped_csv_out = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/annotations/came16_labels_mapped.csv"
annotation_path = "/home/work/mytest/gnn/CPathPatchFeature/camelyon/annotations/came16-labels.csv"

data_name = "came16"

# Check the dataset distribution
patch_counts = []
for f in glob.glob(f"{pt_directory}/*.pt")[:50]:
    payload = torch.load(f, map_location="cpu")
    
    # Check if payload is a dict or a direct Tensor
    if isinstance(payload, dict):
        feats = payload["features"]
    else:
        feats = payload  # Payload is already the features tensor
        
    patch_counts.append(feats.size(0))

pt_files = glob.glob(f"{pt_directory}/*.pt")
print(f"\nFound {len(pt_files)} .pt files")

#print_payload(pt_files)

#print(f"Sampled {len(patch_counts)} slides:")
#print(f"  Min patches: {min(patch_counts)}")
#print(f"  Max patches: {max(patch_counts)}")
#print(f"  Avg patches: {sum(patch_counts) / len(patch_counts):.1f}")

# 1. Build mapped CSV
df_mapped = build_camelyon16_mapped_csv(
    annotation_csv_path=annotation_path,
    pt_dir=pt_directory,
    coord_dir=coord_directory,
    output_csv_path=mapped_csv_out
)
#print(df_mapped.head())

#print("\nMapped CSV columns:")
#print(df_mapped.columns.tolist())

#print("\nMapped CSV sample:")
#print(df_mapped[[
#            "slide_id", "file_path", "coord_path", "label"
#        ]].head()
#)

# Instantiate Train Dataset (tumor_nnn & normal_nnn)
train_dataset = CamelyonPatchFeatureDataset(mapped_csv_path=mapped_csv_out, split='train')
full_train =  len(train_dataset)

sample_train = train_dataset[0]
#print("\n--- Train Sample ---")
#print("Slide ID:", sample_train["slide_id"])
#print("Features shape:", sample_train["features"].shape)
#print("Label:", sample_train["label"].item())

# 3. Instantiate Test Dataset (test_nnn)
test_dataset = CamelyonPatchFeatureDataset(mapped_csv_path=mapped_csv_out, split='test')
#if len(test_dataset) > 0:
#    sample_test = test_dataset[0]
#    print("\n--- Test Sample ---")
#    print("Slide ID:", sample_test["slide_id"])
#    print("Features shape:", sample_test["features"].shape)
#    print("Label:", sample_test["label"].item())
    
# Split into Train, Validation, Test sets
train_size = int(0.85 * len(train_dataset)+0.5)
val_size = int(0.15 * len(train_dataset)+0.5)
test_size = len(test_dataset)

train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

show_labels(train_dataset, "TRAIN")
show_labels(val_dataset, "VAL")
show_labels(test_dataset, "TEST")

# PopTorch Training & Inference Options
opts_train = poptorch.Options()
opts_train.replicationFactor(rept_fac)
opts_train.deviceIterations(device_itr)
opts_train.Training.gradientAccumulation(grad_accum)
opts_train.enableExecutableCaching("./poptorch_cache")

opts_eval = poptorch.Options()
opts_eval.replicationFactor(rept_fac)
opts_eval.deviceIterations(device_itr)
opts_eval.enableExecutableCaching("./poptorch_cache")

for learn_rate in loop_learn_rates:
    for pos_weight in loop_pos_weights:
   
        batch_size = BATCH_SIZE
        train_loader = poptorch.DataLoader(options=opts_train, dataset=train_dataset, batch_size=batch_size,                                                         collate_fn=ipu_collate_fn_train, shuffle=True, drop_last=True)
        val_loader = poptorch.DataLoader(options=opts_eval, dataset=val_dataset, batch_size=batch_size,                                                             collate_fn=ipu_collate_fn_eval, drop_last=True)
        test_loader = poptorch.DataLoader(options=opts_eval, dataset=test_dataset, batch_size=batch_size,                                                            collate_fn=ipu_collate_fn_eval, drop_last=True)

        # Dynamically infer feature dimension from first sample
        sample_feat = train_dataset[0]
        feature_dim = sample_feat["features"].shape[1]
        # print(f"Inferred Feature Dimension: {feature_dim}")

        print(f"\n{'='*90}")
        print(f"✅  Pos Weight={pos_weight}, Learn rate={learn_rate:.2e}")
        print('='*90)

        # Create base feature extractor / GNN
        # 1. Create two separate instances of your GCN-MIL architecture
        base_model_train = WSIPatchGCNMIL(in_channels=feature_dim, hidden_channels=128)
        base_model_eval  = WSIPatchGCNMIL(in_channels=feature_dim, hidden_channels=128)
        model_name = base_model_train.name()
        
        # 2. Synchronize the weights before compilation so they start identical
        base_model_eval.load_state_dict(base_model_train.state_dict())

        # 3. Wrap them into their respective isolated Training/Evaluation wrappers
        train_wrapper = TrainingWrapper(base_model_train, pos_weight=pos_weight)
        eval_wrapper = TrainingWrapper(base_model_eval, pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW(train_wrapper.parameters(), lr=learn_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
        
        # 4. Instantiate PopTorch models safely
        poptorch_train = poptorch.trainingModel(train_wrapper, options=opts_train, optimizer=optimizer)
        print("Training graph compiled successfully!")
        
        poptorch_eval = poptorch.inferenceModel(eval_wrapper, options=opts_eval)
        print("Evaluation graph compiled successfully! Starting validation loop...")

        # Before running PopTorch, test exactly one CPU batch
        bx, bedge, by = next(iter(train_loader))

        #print("bx.shape   :", bx.shape)
        #print("bedge.shape:", bedge.shape)
        #print("by.shape   :", by.shape)

        #print("\nFirst edges:")
        #print(bedge[:, :5])

        # ==============================================================================
        # EXECUTION LOOP
        # ==============================================================================

        print(f"\n☕ Training PopTorch {model_name} on {train_size} train, {val_size} val, and {test_size} test slides...")

        patience_counter = 0
        epochs_without_improvement = 0
        epochs_without_lr_improvement = 0

        best_val_loss = float("inf")
        best_val_f1 = -float("inf")
        save_model_name = f"{model_name}_{data_name}_pw{pos_weight}_lr{learn_rate}.pt"
        SAVE_MODEL_PATH = os.path.join(SAVE_MODEL_DIR, save_model_name)

        for epoch in range(1, EPOCHS+1):
            
            poptorch_train.train()
            total_train_loss = 0.0

            before = base_model_train.classifier.weight.detach().clone()

            # --- TRAINING PASS ---
            for bx, bedge, by in train_loader:
            # Now bx, bedge, by are already PyTorch Tensors unpacked from the tuple!
                _, loss = poptorch_train(bx, bedge, by)
                total_train_loss += loss.mean().item()

            # Pull trained IPU weights back to base_model on Host CPU/Release IPU hardware for validation
            poptorch_train.copyWeightsToHost()
            
            after = base_model_train.classifier.weight.detach().clone()
            weight_change = ((after - before).abs().mean().item())
            
            poptorch_train.detachFromDevice()

            if(len(train_loader) > 0):
                avg_train_loss = total_train_loss / len(train_loader)
            else:
                print("Divided by zero, len(train_loader) = 0 ")
                break

            # CRITICAL CORRECTION: SINK TRAINED WEIGHTS INTO EVAL MODEL BEFORE VALIDATION
            base_model_eval.load_state_dict(base_model_train.state_dict())
            
            # --- VALIDATION PASS ---
            
            if epoch > 1:
                poptorch_eval.attachToDevice()
                poptorch_eval.copyWeightsToDevice()
                
            val_metrics = evaluate(poptorch_eval, val_loader, verbose=False)

            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            # Release IPU hardware back to training loop
            poptorch_eval.detachFromDevice()

            current_lr = optimizer.param_groups[0]["lr"]
            #print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | \
            #      Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Learn_rate: {current_lr:.8f}")
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | \
                  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Learn_rate: {current_lr:.2e} | Weight Δ: {weight_change:.6f}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"✅  Saved best model to {SAVE_MODEL_PATH}")
                torch.save(base_model_eval.state_dict(), SAVE_MODEL_PATH)

                epochs_without_lr_improvement = 0
                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1
                epochs_without_lr_improvement += 1

            if epochs_without_lr_improvement >= lr_patience:
                new_lr = max(current_lr * lr_factor, min_lr)
                if new_lr < current_lr:
                    current_lr = new_lr
                    print(f"Reducing LR to {current_lr:.2e}")
                    for group in optimizer.param_groups:
                        group["lr"] = current_lr
                    poptorch_train.setOptimizer(optimizer)
                    epochs_without_lr_improvement = 0

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

        # Load Best Weights for Final Evaluation
        print(f"✅  Loading best model as test model from {SAVE_MODEL_PATH}")
        base_model_eval.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location="cpu"))
        
        # synchronize loaded host checkpoint -> inference IPU
        poptorch_eval.attachToDevice()
        poptorch_eval.copyWeightsToDevice()

        print("\nRunning Final Test Evaluation...")
        test_start = time.time()
        test_metrics = evaluate(poptorch_eval, test_loader, split_name="Inference", verbose=True)
        test_elapsed = time.time() - test_start
        test_loss = test_metrics["loss"]
        test_acc = test_metrics["accuracy"]
        print(
            f"\nTest Loss: {test_metrics['loss']:.4f} | "
            f"Accuracy: {test_metrics['accuracy']:.4f} | "
            f"Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"ROC-AUC: {test_metrics['roc_auc']:.4f} | "
            f"CM: {test_metrics['confusion_matrix']}"
        )

        
        # save the result in the csv file
        CSV_FILE = os.path.join(SAVE_RESULT_DIR, "GCN_vs_GNN_benchmark.csv")
        tn, fp, fn, tp = test_metrics['confusion_matrix'].ravel()
        pd.DataFrame([{
            "model name": model_name,
            "dataset": data_name,
            "train_set": train_size,
            "batch size": batch_size,
            "learn rate": learn_rate,
            "repli factor": rept_fac,
            "device itr": device_itr,
            "grad accum": grad_accum,
            "patience": patience,
            "lr_patience" : lr_patience,
            "accuracy": f"{test_metrics['accuracy']:.4f}",
            "precision": f"{test_metrics['precision']:.4f}",
            "recall": f"{test_metrics['recall']:.4f}",
            "f1": f"{test_metrics['f1']:.4f}",
            "roc_auc": f"{test_metrics['roc_auc']:.4f}",
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "images_per_sec": f'{(test_size / test_elapsed):.1f}',
        }]).to_csv(CSV_FILE, mode="a", header=not os.path.exists(CSV_FILE), index=False)
                
        poptorch_eval.detachFromDevice()
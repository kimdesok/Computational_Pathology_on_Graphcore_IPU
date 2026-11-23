import torch
import torch.nn as nn
import torch.nn.functional as F

import poptorch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ==============================================================================
# 1. GENERATE SYNTHETIC WSI PATCH-GRAPH DATASET
# ==============================================================================
def create_grid_graph(grid_size=4, num_features=8, is_tumor=False):
    num_nodes = grid_size * grid_size
    x = torch.randn((num_nodes, num_features), dtype=torch.float32)
    label = 1 if is_tumor else 0
    if is_tumor:
        x += 2.5

    edge_list = []
    for r in range(grid_size):
        for c in range(grid_size):
            node_id = r * grid_size + c
            # Self-loops explicitly included in edge list to keep static shapes
            edge_list.append([node_id, node_id])
            if c + 1 < grid_size:
                neighbor_id = r * grid_size + (c + 1)
                edge_list.append([node_id, neighbor_id])
                edge_list.append([neighbor_id, node_id])
            if r + 1 < grid_size:
                neighbor_id = (r + 1) * grid_size + c
                edge_list.append([node_id, neighbor_id])
                edge_list.append([neighbor_id, node_id])

    # Use int32 explicitly for IPU compatibility
    edge_index = torch.tensor(edge_list, dtype=torch.int32).t().contiguous()
    y = torch.tensor([label], dtype=torch.int32)

    return Data(x=x, edge_index=edge_index, y=y)

print("Generating synthetic WSI patch-graphs...")
raw_dataset = [create_grid_graph(grid_size=4, num_features=8, is_tumor=(i % 2 == 1)) for i in range(320000)]

def ipu_pyg_collate(data_list):
    """
    Collates PyG Data objects into fixed 3D/2D tensors 
    where Dimension 0 is always batch_size.
    """
    x = torch.stack([d.x for d in data_list], dim=0)                     # [B, N, F]
    edge_index = torch.stack([d.edge_index for d in data_list], dim=0)   # [B, 2, E] (int32)
    y = torch.stack([d.y for d in data_list], dim=0)                     # [B, 1] (int32)

    return x, edge_index, y

# ==============================================================================
# 2. DEFINE MODEL TO HANDLE BATCH-FIRST GRAPH TENSORS
# ==============================================================================
class PatchGNN(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=8, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False, normalize=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False, normalize=False)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        batch_size, num_nodes, num_features = x.shape
        
        # Flatten batch for GCN operations
        x_flat = x.view(batch_size * num_nodes, num_features)
        
        # Adjust edge indices with offsets for disjoint graph batching
        offsets = (torch.arange(batch_size, device=x.device, dtype=torch.int32) * num_nodes).view(batch_size, 1, 1)
        edge_index_flat = (edge_index + offsets).transpose(0, 1).reshape(2, -1)
        
        # Static Batch Vector for Global Pooling
        batch_vector = torch.arange(batch_size, device=x.device, dtype=torch.int32).view(-1, 1).expand(batch_size, num_nodes).reshape(-1)

        # Static Message Passing
        h = self.conv1(x_flat, edge_index_flat)
        h = F.relu(h)
        h = self.conv2(h, edge_index_flat)
        h = F.relu(h)

        # FIX: Pass size=batch_size explicitly to avoid internal int(index.max())
        pooled = global_mean_pool(h, batch_vector, size=batch_size)
        out = self.fc(pooled)
        return out

class TrainingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, edge_index, y):
        logits = self.model(x, edge_index)
        # CrossEntropyLoss accepts int64 targets on CPU/IPU
        loss = self.loss_fn(logits, y.view(-1).long())
        return logits, loss

# ==============================================================================
# 3. POPTORCH & BOW-2000 HARDWARE CONFIGURATION
# ==============================================================================
opts = poptorch.Options()

opts.replicationFactor(4)                      # 4 IPU Processors
opts.deviceIterations(1)                       # 10 loops per step
opts.Training.gradientAccumulation(4)           # 4 accumulation steps

micro_batch_size = 1

dataloader = poptorch.DataLoader(
    options=opts,
    dataset=raw_dataset,
    batch_size=micro_batch_size,
    collate_fn=ipu_pyg_collate,
    shuffle=True,
    drop_last=True
)

# ==============================================================================
# 4. INITIALIZE POPTORCH ENGINE & TRAIN
# ==============================================================================
base_model = PatchGNN(in_channels=8, hidden_channels=16, num_classes=2)
wrapped_model = TrainingWrapper(base_model)
optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.01)

poptorch_model = poptorch.trainingModel(
    wrapped_model, 
    options=opts, 
    optimizer=optimizer
)

print("\nCompiling graph and starting GNN Training on 8 IPUs...")
poptorch_model.train()

for epoch in range(1, 10):
    total_loss = 0.0
    steps = 0
    
    for batch_x, batch_edge_index, batch_y in dataloader:
        logits, loss = poptorch_model(batch_x, batch_edge_index, batch_y)
        total_loss += loss.mean().item()
        steps += 1

    print(f"Epoch {epoch:02d} | Avg Loss: {total_loss / steps:.4f}")

poptorch_model.detachFromDevice()
print("\nTraining completed successfully on 8 IPUs!")
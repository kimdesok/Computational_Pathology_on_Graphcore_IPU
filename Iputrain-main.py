#!/usr/bin/env python
# coding: utf-8
#
# One script, two jobs:
#   MODE = "train"   -> runs on the IPU box, trains the model, saves a .pt file
#   MODE = "compile" -> runs on the NPU box, loads that .pt file, compiles it
#                       with RBLN SDK, and runs the test-set accuracy check
#
# Just flip MODE depending on which machine you're on.


import subprocess, json, threading, time
import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
import poptorch

from torchvision import transforms
from datasets import load_dataset, load_from_disk
from collections import Counter
from torchvision.models import VisionTransformer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,\
                roc_auc_score, confusion_matrix
import pandas as pd
import timm
import gc
import os
os.environ["POPTORCH_LOG_LEVEL"] = "ERR"


RESNET_MODELS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}

import warnings

# Suppress multiprocessing resource tracker warnings
os.environ["PYTHONWARNINGS"] = "ignore:resource_tracker:UserWarning"
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Hide the specific PyTorch positional args warning
warnings.filterwarnings("ignore", category=UserWarning, message="positional args are being deprecated")

import poptorch
import sys
#sys.path.insert(0, "/opt/pytorch/lib/python3.8/site-packages")

import itertools

# ---------------------------------------------------------------------
# Stuff both modes need: dataset, transform, model definition
# ---------------------------------------------------------------------

class Camelyon17Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample["image"].convert("RGB")
        label = torch.tensor(sample["label"], dtype=torch.long)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def create_resnet(model_name, num_classes=2, pretrained=True):
    if model_name not in RESNET_MODELS:
        raise ValueError(f"Unsupported ResNet: {model_name}")
    constructor, weights = RESNET_MODELS[model_name]
    backbone = constructor(weights=weights if pretrained else None)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return backbone


class ResNetCamelyon(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = create_resnet(model_name, num_classes, pretrained)

    def forward(self, x, labels=None):
        x = x.to(self.backbone.conv1.weight.dtype)
        logits = self.backbone(x)

        if labels is None:
            return logits
        return logits, F.cross_entropy(logits.float(), labels)



class ResNetCamelyonIPU(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=2,
                 pretrained=True, num_stages=4):
        super().__init__()
        self.backbone = create_resnet(model_name, num_classes, pretrained)
        self.num_stages = num_stages

        # Flatten residual groups without registering duplicate modules
        b = self.backbone
        self.blocks = tuple(
            block for layer in (b.layer1, b.layer2, b.layer3, b.layer4)
            for block in layer
        )

        # Partition residual blocks according to architecture
        self.stage_blocks = {
            "resnet18": {
                4: [2, 2, 2, 2],
                8: [1, 1, 1, 1, 1, 1, 1, 1]
            },
            "resnet34": {
                4: [3, 4, 6, 3],
                8: [2, 2, 2, 2, 2, 2, 2, 2]
            },
            "resnet50": {
                4: [3, 4, 6, 3],
                8: [2, 2, 2, 2, 2, 2, 2, 2]
            }
        }

        if model_name not in self.stage_blocks:
            raise ValueError(f"Unsupported model: {model_name}")
        if num_stages not in self.stage_blocks[model_name]:
            raise ValueError("num_stages must be 4 or 8")

        self.partition = self.stage_blocks[model_name][num_stages]

        assert len(self.partition) == num_stages
        assert sum(self.partition) == len(self.blocks)

    def forward(self, x, labels=None):
        b = self.backbone
        block_idx = 0

        for stage, n_blocks in enumerate(self.partition):
            with poptorch.Block(f"Stage_{stage}", ipu_id=stage):

                # First stage: ResNet stem
                if stage == 0:
                    x = b.conv1(x)
                    x = b.bn1(x)
                    x = b.relu(x)
                    x = b.maxpool(x)

                # Residual blocks assigned to this IPU
                for _ in range(n_blocks):
                    x = self.blocks[block_idx](x)
                    block_idx += 1

                # Last stage: classification head
                if stage == self.num_stages - 1:
                    x = b.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = x.to(b.fc.weight.dtype)
                    
                    #print(f"FC input dtype: {x.dtype}")
                    #print(f"FC weight dtype: {b.fc.weight.dtype}")
                    #print(f"FC bias dtype: {b.fc.bias.dtype}")

                    logits = b.fc(x)

                    if labels is not None:
                        loss = F.cross_entropy(logits.float(), labels)

        if labels is None:
            return logits
        return logits, loss



class ViTCamelyon(nn.Module):
    def __init__(self, model_name=None, num_classes=2, pretrained=True, drop_path_rate=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate
        )
    
    def forward(self, x, labels=None):
        x = x.to(self.backbone.patch_embed.proj.weight.dtype)
        logits = self.backbone(x)

        if labels is None:
            return logits

        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        return logits, loss


class ViTCamelyonIPU(nn.Module):

    def __init__(self, model_name=None, num_classes=2, pretrained=True, num_stages=4, drop_path_rate=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes,
            drop_path_rate=drop_path_rate
        )
        self.num_stages = num_stages
        self.stage_blocks = {
            4: [3, 3, 3, 3],
            8: [2, 2, 2, 2, 1, 1, 1, 1]
        }
        
        if num_stages not in self.stage_blocks:
            raise ValueError("num_stages must be 4 or 8")
        
        assert len(self.stage_blocks[num_stages]) == num_stages
        assert sum(self.stage_blocks[num_stages]) == len(self.backbone.blocks)
            
        if sum(self.stage_blocks[num_stages]) != len(self.backbone.blocks):
            raise ValueError("Stage partition does not match transformer depth")

    
    def forward(self, x, labels=None):
        block_idx = 0

        for stage, n_blocks in enumerate(self.stage_blocks[self.num_stages]):
            with poptorch.Block(f"Stage_{stage}", ipu_id=stage):
                if stage == 0:
                    x = x.to(self.backbone.patch_embed.proj.weight.dtype)
                    x = self.backbone.patch_embed(x)
                    x = self.backbone._pos_embed(x)
                    x = self.backbone.norm_pre(x)

                for _ in range(n_blocks):
                    x = self.backbone.blocks[block_idx](x)
                    block_idx += 1

                if stage == self.num_stages - 1:
                    x = self.backbone.norm(x)
                    logits = self.backbone.forward_head(x)

                    if labels is not None:
                        loss = torch.nn.functional.cross_entropy(
                            logits.float(), labels
                        )
                        loss = poptorch.identity_loss(loss, reduction="mean")

        if labels is None:
            return logits
        return logits, loss

    

def create_camelyon_model(model_name, parallel_mode,
                          num_classes=2, pretrained=True):

    if model_name.startswith("resnet"):
        if parallel_mode == "data":
            return ResNetCamelyon(model_name, num_classes, pretrained)

        num_stages = {
            "model0": 4,
            "model1": 8,
            "hybrid0": 4,
            "hybrid1": 4,
        }.get(parallel_mode)

        if num_stages is None:
            raise ValueError(f"Unsupported mode: {parallel_mode}")

        return ResNetCamelyonIPU(
            model_name, num_classes, pretrained, num_stages
        )

    raise ValueError(f"Unsupported model: {model_name}")

def make_random_subset(hf_dataset, n_samples=None, seed=42):
    shuffled = hf_dataset.shuffle(seed=seed)
    if n_samples is None:
        return shuffled
    return shuffled.select(range(min(n_samples, len(shuffled))))


def load_camelyon(dataset_name, dataset_path):
    if os.path.exists(dataset_path) :
        print("Loading dataset from local disk...")
        return load_from_disk(dataset_path)
    print("Local data not found. Downloading from Hugging Face...")
    ds = load_dataset(dataset_name, cache_dir=dataset_path)
    ds.save_to_disk(dataset_path)
    return ds

# ---------------------------------------------------------------------
# MODE == "train"  -> IPU training (poptorch)
# ---------------------------------------------------------------------
def cleanup_poptorch(*models):
    for name, obj in models:
        if obj is None:
            continue

        try:
            obj.detachFromDevice()
            #print(f"{name}: detached")
        except Exception as e:
            print(f"{name}: detach skipped: {e}")

        try:
            obj.destroy()
            #print(f"{name}: destroyed")
        except Exception as e:
            print(f"{name}: destroy skipped: {e}")

    gc.collect()

    
def sync_val_weights():
    poptorch_model.copyWeightsToHost()
    val_model.load_state_dict(model.state_dict())
    poptorch_val.copyWeightsToDevice()
    # first time through: poptorch_val hasn't compiled yet, so there's
    # nothing to push into -- the upcoming evaluate_ipu() call below will
    # compile it fresh, using the weights we just loaded into val_model

def evaluate_ipu(poptorch_infer_model, data_loader):
    total, correct, total_loss, n_steps = 0, 0, 0.0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.half()
            logits = poptorch_infer_model(images)
            logits = logits.float()
            
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            n_steps += 1
    return total_loss / n_steps, correct / total

# Add a background power monitor

import threading, time
import numpy as np
import subprocess, json

def get_ipu_utilization():
    try:
        result = subprocess.run(["gc-monitor", "-j"],
                                capture_output=True, text=True, timeout=5, check=True)
        data = json.loads(result.stdout)

        ipus = [ipu for card in data.get("cards", [])
                    for ipu in card.get("ipus", [])]
        ipus.sort(key=lambda x: int(x["ID"]))

        utils = [float(str(ipu.get("IPU utilisation", "0")).replace("%", "").strip() or 0)
                 for ipu in ipus]
        powers = [float(str(ipu.get("IPU Power", "0")).replace("W", "").strip() or 0)
                  for ipu in ipus]
        
        #print("IPU utilization:", utils)
        #print("IPU power (W): ", powers)
        #print("Total IPU power:", sum(powers), "W")

        return utils, powers

    except Exception as e:
        print(f"IPU monitoring error: {e}")
        return [], []    
    
class IPUPowerMonitor:
    def __init__(self, interval=1.0, n_ipus=8):
        self.interval = interval
        self.n_ipus = n_ipus
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def _monitor(self):
        while not self.stop_event.is_set():
            utils, powers = get_ipu_utilization()
            if len(powers) >= self.n_ipus:
                total_power = sum(powers[:self.n_ipus])
                self.samples.append((time.perf_counter(), total_power))
            self.stop_event.wait(self.interval)

    def start(self):
        self.samples = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def summary(self):
        return float(np.mean([p for t, p in self.samples])) if self.samples else float("nan")

def get_data(dataset_path, cam_dataset, my_transform, rep_fact, device_itr, grad_accum, batch_size, parallel_mode):
    
    #Dataset loading
    dataset = load_camelyon(cam_dataset, dataset_path)
    # List the possible names in order of preference
    possible_splits = ['validation', 'valid']
    
    # Find the first one that actually exists in the dataset
    valid_key = next((s for s in possible_splits if s in dataset), None)
    
    if valid_key:
        val_data = dataset[valid_key]
    else:
        print("No validation split found.")

    print("No of data:", len(dataset['train']), len(dataset[valid_key]), len(dataset['test']))

    full_train = len(dataset['train']) 
    full_test = len(dataset['test'])
    full_valid = len(dataset[valid_key])

    # Train data prep.
    train_hf = make_random_subset(dataset["train"], n_samples=full_train, seed=42)
    train_data = Camelyon17Dataset(train_hf, transform=my_transform)
    print("train labels:", Counter(train_hf["label"]))
    #print(f"Rep. Factor: {rep_fact}")

    # Validation data prep
    val_hf = make_random_subset(dataset[valid_key], n_samples=full_valid, seed=43)
    print("val labels:", Counter(val_hf["label"]))
    val_data = Camelyon17Dataset(val_hf, transform=my_transform)

    # Test data preparation
    test_hf = make_random_subset(dataset["test"], n_samples=full_test, seed=44)
    print("test labels:", Counter(test_hf["label"]))
    test_data = Camelyon17Dataset(test_hf, transform=my_transform)
    
    # Train options prep.
    train_opts = poptorch.Options()  
    train_opts.replicationFactor(rep_fact)
    train_opts.deviceIterations(device_itr)
    train_opts.Training.gradientAccumulation(grad_accum)

    train_opts.Precision.enableStochasticRounding(True)
    train_opts.Precision.setPartialsType(torch.half)
                                         
    # Apply the same stage memory allocations and caching
    num_ipus = 4
    if parallel_mode != 'model0':
        num_ipus = 8
    amp = {f"IPU{i}": 0.2 for i in range(num_ipus)}                                     
    train_opts.setAvailableMemoryProportion(amp)
    train_opts.enableExecutableCaching("./poptorch_cache")

    # Validation options prep.
    valid_opts = poptorch.Options()
    valid_opts.replicationFactor(rep_fact)  # Keep identical to training layout
    valid_opts.deviceIterations(device_itr) 

    # FP16 Validation Precision
    valid_opts.Precision.setPartialsType(torch.half)    # Keep ON for numerical stability
    valid_opts.setAvailableMemoryProportion(amp)
    valid_opts.enableExecutableCaching("./poptorch_cache")
    
    if parallel_mode in ("model0", "model1", "hybrid0", "hybrid1"):
        train_opts.setExecutionStrategy(poptorch.PipelinedExecution())
        valid_opts.setExecutionStrategy(poptorch.PipelinedExecution())

    train_loader = poptorch.DataLoader(
                        train_opts, train_data, batch_size=batch_size, shuffle=True, drop_last=True, 
                        mode=poptorch.DataLoaderMode.Async, async_options={"buffer_size": 2}, 
                        num_workers=8, persistent_workers=True
                    )

    val_loader = poptorch.DataLoader(
        valid_opts, val_data, batch_size=batch_size, shuffle=False, drop_last=True,
        mode=poptorch.DataLoaderMode.Async,
        num_workers=4, persistent_workers=True
    )

    test_loader = poptorch.DataLoader(
        valid_opts, test_data, batch_size=batch_size, shuffle=False, drop_last=True, 
        mode=poptorch.DataLoaderMode.Async,
        num_workers=4, persistent_workers=True
    )

    return train_loader, val_loader, test_loader, train_opts, valid_opts 


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

def get_models(model_name, parallel_mode, drop_path_rate=0.1):
    stage_map = {
        "model0": 4,
        "model1": 8,
        "hybrid0": 4,
        "hybrid1": 4
    }

    if parallel_mode not in (*stage_map, "data"):
        raise ValueError(f"Unsupported parallel mode: {parallel_mode}")

    def build_model(pretrained):
        if model_name.startswith("resnet"):
            if parallel_mode == "data":
                return ResNetCamelyon(
                    model_name, num_classes=2, pretrained=pretrained
                )
            return ResNetCamelyonIPU(
                model_name, num_classes=2, pretrained=pretrained,
                num_stages=stage_map[parallel_mode]
            )

        elif model_name.startswith("vit"):
            if parallel_mode == "data":
                return ViTCamelyon(
                    model_name, num_classes=2, pretrained=pretrained,
                    drop_path_rate=drop_path_rate
                )
            return ViTCamelyonIPU(
                model_name, num_classes=2, pretrained=pretrained,
                num_stages=stage_map[parallel_mode],
                drop_path_rate=drop_path_rate
            )

        raise ValueError(f"Unsupported model: {model_name}")

    model = build_model(pretrained=True)
    val_model = build_model(pretrained=False)
    test_model = build_model(pretrained=False)

    return model, val_model, test_model


def plot_training(history, save_path, title_text="Loss and Accuracy"):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title_text, fontsize=14)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_ylim(0.0, 0.7)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()

    axes[2].plot(epochs, history["lr"])
    axes[2].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
    axes[2].set_ylim(1e-8, 1e-3)
    axes[2].set_yscale("log")

    for ax in axes: ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    
    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    save_path = str(save_path).replace(".png", ".csv")
    #print(save_path)
    df.to_csv(save_path, index=False)
   
    
import subprocess
import itertools

import os
import sys
import time
import pandas as pd
import torch
import poptorch
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from datetime import datetime


# Note: Ensure transforms, get_data, get_models, evaluate_ipu, 
# IPUPowerMonitor, and cleanup_poptorch are imported/defined in your environment scope.

def main():
    # -------------------------------------------
    #  Read active loop variables from environment
    # -------------------------------------------

    batch_size = int(os.environ.get("RUN_BATCH_SIZE", 1))
    learn_rate = float(os.environ.get("RUN_LEARN_RATE", 1e-6))
    
    model_name = os.environ.get("RUN_MODEL_NAME", "vit_tiny_patch16_224")
    parallel_mode = os.environ.get("RUN_PARALLEL_MODE", "model0")
    aug_dp = os.environ.get("RUN_AUG_DP", "baseline")
    
    rep_fact = int(os.environ.get("RUN_REP_FACT", 1))
    device_itr = int(os.environ.get("RUN_DEVICE_ITR", 4))
    grad_accum = int(os.environ.get("RUN_GRAD_ACCUM", 32))
                                    
    expr_type = os.environ.get("RUN_EXPR_TYPE", "came16")
    dataset_path = os.environ.get("RUN_DATA_PATH", "/home/work/datasets/camelyon16_1aurent")
    dataset_name = os.environ.get("RUN_DATA_NAME", "1aurent/PatchCamelyon")

    MODE = "train"

    # Constants
    num_epochs = 100
    current_decay = 5e-2
    patience = 10
    lr_patience = 3
    lr_factor = 0.5
    min_lr = 1e-9
    drop_path_rate = 0.3
    warmup_epochs = 10  # Ramps up learning rate over the first 10 epochs
    
    # Define the base transformations in a standard list
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 180)),
    ]

    # Use a regular if statement to conditionally add the transformation
    if aug_dp in ["color_jitter", "color_drop_path"]:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02
            )
        )

    # Add the final transformations
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    my_transform = transforms.Compose(transform_list)

    print("PopTorch:", poptorch.__file__)
    print("Version:", getattr(poptorch, "__version__", "unknown"))
                              
    print(f"\n{('-'*100)}")
    print(f"🚀  Executing Worker Pass: Dataset Type={expr_type} | Model={model_name} | Parallel mode={parallel_mode} |\n"
          f"   Batch size={batch_size} | Learn rate={learn_rate} | Augment/Drop_path={aug_dp}@{drop_path_rate} |\n"
          f"   Replication fact={rep_fact} | Device itr={device_itr} | Gradient accum={grad_accum} |")
    print('-'*100)
    
    short_name = "_".join(model_name.split("_")[:2])
    
    os.makedirs(expr_type, exist_ok=True)
    
    print(f"**********  Experiment Type in the main: {expr_type} *********")
    print(f"Dataset to load: {dataset_name} available at {dataset_path}")
    
    dataset_type = dataset_path.split("/")[-1]
    short_dataset_name = dataset_name.split("/")[-1]

    train_loader, val_loader, test_loader, train_opts, valid_opts = get_data(
        dataset_path, dataset_name, my_transform, rep_fact, device_itr, grad_accum, batch_size, parallel_mode
    )
    
    full_train = len(train_loader.dataset)
    print("No. of patches used for training =", full_train)
    
    if aug_dp in ["baseline", "color_jitter"]:
        drop_path_rate = 0.0
      
    model, val_model, test_model = get_models(model_name, parallel_mode, drop_path_rate)
    model.train()
    freeze_bn(model)

    current_lr = learn_rate
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=current_decay)
    optimizer = poptorch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=current_decay)

    poptorch_model = poptorch.trainingModel(model, options=train_opts, optimizer=optimizer)
    
    val_model.load_state_dict(model.state_dict())
    val_model.eval()

    poptorch_val = poptorch.inferenceModel(val_model, options=valid_opts)
    val_has_compiled = False
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epochs_without_lr_improvement = 0

    # training loss curves
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    
    my_time_stamp = datetime.now().strftime("%d-%m-%Y_%H:%M")
    print("Time:", my_time_stamp)
    
    # Use the same ID everywhere: model paths in expr_type and .csv & .png files in 'curves' directories, and benchmark.csv
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # to generate the file name
    
    # start with the checkpoint
    checkpoint_dir = Path(f"/home/work/mytest/Graphcore_IPU/{expr_type}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = Path(f"{short_name}_bs{batch_size}_lr{learn_rate}_rf{rep_fact}"\
    f"_di{device_itr}_ga{grad_accum}_{short_dataset_name}_{time_stamp}")
    
    checkpoint_path = checkpoint_dir/f"{run_id}.pt"
    
    CURVE_DIR = checkpoint_dir/Path("curves")
    os.makedirs(str(CURVE_DIR), exist_ok=True)  # saving the loss curves
 
    start_time = time.time()
    
    print(f"🔥 Starting training with a {warmup_epochs}-epoch linear warmup to target LR: {learn_rate:.2e}")

    # ----------------------------------------------------
    # Core Training Loop Execution
    # ----------------------------------------------------
    for epoch in range(num_epochs):
        running_loss = 0.0 
        n_steps = 0
        time0 = time.time()

        # Learning Rate Warmup Logic
        if epoch < warmup_epochs:
            # Linearly scale LR from min_lr up to the target learn_rate
            current_lr = min_lr + (learn_rate - min_lr) * ((epoch + 1) / warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            # Tell PopTorch to update the compiled device engine with the new warmup LR
            poptorch_model.setOptimizer(optimizer)
            if epoch == 0 or (epoch + 1) == warmup_epochs:
                print(f" [Warmup Stage] Setting learning rate to {current_lr:.2e}")

        poptorch_model.train()

        for images, labels in train_loader:
            images = images.half()
            logits, loss = poptorch_model(images, labels)
            running_loss += loss.mean().item()
            n_steps += 1

        train_loss = running_loss / n_steps

        poptorch_model.copyWeightsToHost()
        val_model.load_state_dict(model.state_dict())
        poptorch_model.detachFromDevice()
   
        if val_has_compiled:
            poptorch_val.attachToDevice()
            poptorch_val.copyWeightsToDevice()
            
            
        # evaluate the training model's checkpoint directly on the CPU, bypassing PopTorch entirely 
        if model_name.startswith("resnet"):
            cpu_model = ResNetCamelyon(model_name, pretrained=False)
            cpu_model.load_state_dict(val_model.state_dict())
            cpu_model.float().eval()

            images, labels = next(iter(val_loader))

            with torch.no_grad():
                logits_cpu = cpu_model(images.float())
                logits_ipu = poptorch_val(images.half()).float()

                loss_cpu = F.cross_entropy(logits_cpu, labels)
                loss_ipu = F.cross_entropy(logits_ipu, labels)

            #print("CPU loss:", loss_cpu.item())
            #print("IPU loss:", loss_ipu.item())
            #print("Max logit diff:", (logits_cpu-logits_ipu).abs().max().item())

        val_loss, val_acc = evaluate_ipu(poptorch_val, val_loader)
        val_has_compiled = True

        poptorch_val.detachFromDevice()
        poptorch_model.attachToDevice() 

        epoch_time = time.time() - time0

        print(
            f"Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | elapsed={epoch_time:.1f} sec"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            epochs_without_lr_improvement = 0

            print(f"✅ Saved best model to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            # Only count plateaus toward decaying the learning rate AFTER warmup is over
            if epoch >= warmup_epochs:
                epochs_without_lr_improvement += 1

        if epochs_without_lr_improvement >= lr_patience:
            new_lr = max(current_lr * lr_factor, min_lr)
            if new_lr < current_lr:
                current_lr = new_lr
                print(f"📉  Plateau detected. Reducing LR to {current_lr:.2e}")
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                poptorch_model.setOptimizer(optimizer)
                epochs_without_lr_improvement = 0
                
        # loss & accuracy curve data
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        title_text = f"{short_name}_{parallel_mode}_bs{batch_size}_lr{learn_rate}_dp{aug_dp}_{run_id}"
        save_path = CURVE_DIR/Path(f"{title_text}.png")
        plot_training(history, save_path, title_text)

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break
            
    print(f"\nTotal train time: {(time.time()-start_time)/60.0:.1f} min")
    poptorch_model.copyWeightsToHost()

    try:
        poptorch_model.detachFromDevice()
        poptorch_model.destroy()
    except Exception as e:
        print("cleanup of train warning:", e)
    
    try:
        poptorch_val.destroy()
    except Exception as e:
        print("cleanup of val warning:", e)

    # ---------------------------------------------------------------------
    # Final Evaluation & Benchmark Pass
    # ---------------------------------------------------------------------
    print(f"✅ Loading best model as test model from {checkpoint_path}")
    test_model.load_state_dict(torch.load(checkpoint_path))
    test_model.eval()

    infer_model = poptorch.inferenceModel(test_model, options=valid_opts)
    predictions, probability, true_labels = [], [], []

    images, labels = next(iter(test_loader))
    with torch.no_grad():
        _ = infer_model(images.half())

    num_stages = 8 if parallel_mode == "model1" else (1 if parallel_mode == "data" else 4)
    n_ipus = num_stages * rep_fact
    monitor = IPUPowerMonitor(interval=1.0, n_ipus=n_ipus)
    no_of_patches = 0

    monitor.start()
    test_start = time.perf_counter()

    try:
        with torch.no_grad():
            for images, labels in test_loader:
                logits = infer_model(images.half()).float()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().tolist())
                probability.extend(probs[:, 1].cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                no_of_patches += labels.numel()
    finally:
        test_elapsed = time.perf_counter() - test_start
        monitor.stop()

    avg_power = monitor.summary()
    throughput = no_of_patches / test_elapsed
    energy_joules = avg_power * test_elapsed
    images_per_joule = throughput / avg_power if avg_power > 0 else float("nan")

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted")
    recall = recall_score(true_labels, predictions, average="weighted")
    f1 = f1_score(true_labels, predictions, average="weighted")
    roc_auc = roc_auc_score(true_labels, probability)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()

    print("------------------------------------------------------------")
    print(f"IPU test accuracy  : {accuracy:.4f} | throughput : {throughput:.2f} images/sec")
    print("------------------------------------------------------------")

    CSV_FILE = f"./{expr_type}/ipu_vs_npu_benchmark.csv"
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    pd.DataFrame([{
        "model name": str(checkpoint_path), "dataset": dataset_type, "train_set": full_train,
        "batch size": batch_size, "learn rate": learn_rate, "repli factor": rep_fact,
        "device itr": device_itr, "grad accum": grad_accum, "parallel mode": parallel_mode, "aug_dp":aug_dp,
        "patience": patience, "lr_patience": lr_patience, "accuracy": f"{accuracy:.4f}",
        "precision": f"{precision:.4f}", "recall": f"{recall:.4f}", "f1": f"{f1:.4f}",
        "roc_auc": f"{roc_auc:.4f}", "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "no_of_patches": no_of_patches, "inference_sec": f"{test_elapsed:.2f}",
        "images_per_sec": f"{throughput:.2f}", "avg_ipu_power_w": f"{avg_power:.2f}",
        "energy_joules": f"{energy_joules:.2f}", "images_per_joule": f"{images_per_joule:.2f}",
        "power_samples": len(monitor.samples),
    }]).to_csv(CSV_FILE, mode="a", header=not os.path.exists(CSV_FILE), index=False)

    import gc
    import sys
    
    # Controlled, Ordered Cleanup Sequence
    print("🧹 Starting controlled resource cleanup...")
    
    # Destroy PopTorch compiled graph wrappers
    for p_model in [poptorch_model, poptorch_val, infer_model]:
        try:
            p_model.detachFromDevice()
        except Exception:
            pass # Skipped if already detached or never attached
        try:
            p_model.destroy()
        except Exception as e:
            pass

    # Delete base PyTorch models
    del model, val_model, test_model, infer_model, poptorch_model, poptorch_val

    # Delete data loaders to shut down background multiprocessing workers (& semaphores)
    del train_loader, val_loader, test_loader, train_opts, valid_opts
    
    gc.collect()

    print("✅ Cleanup complete. Exiting cleanly.")
    sys.exit(0)

if __name__ == "__main__":
    
    # Your training, validation, and testing code here...
    main()
    from datetime import datetime
    my_time_stamp = datetime.now().strftime("%d-%m-%Y_%H:%M")
    print(my_time_stamp)  
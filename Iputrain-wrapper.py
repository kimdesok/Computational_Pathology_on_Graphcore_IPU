#!/usr/bin/env python
# coding: utf-8
#
# One script, two jobs:
#   MODE = "train"   -> runs on the IPU box, trains the model, saves a .pt file
#   MODE = "compile" -> runs on the NPU box, loads that .pt file, compiles it
#                       with RBLN SDK, and runs the test-set accuracy check
#
# Just flip MODE depending on which machine you're on.


import subprocess
import itertools
import os
import shutil

def main():
    # datasets
    dataset_configs = [
        ("came17",  "/home/work/datasets/camelyon17_wilds", "wltjr1007/Camelyon17-WILDS"),
        ("came16",  "/home/work/datasets/camelyon16_1aurent", "1aurent/PatchCamelyon"),
    ]
    # Hyperparameters Grid Search
    loop_batch_sizes = [1]
    loop_learn_rates = [5e-8, 1e-08]  # monitor for 15 epochs and little improvement warrants large rates:5e-6
    loop_model_names = ["vit_small_patch16_224"] #["vit_tiny_patch16_224", 
    loop_aug_dps = ["color_drop_path"] #["color_jitter", "drop_path", "color_drop_path"]
    
    # Clean IPU Profile Configuration Pairing
    ipu_configs = [
        ("model0", 1, 4, 32), 
        ("model1", 1, 8, 32),
        ("hybrid0", 2, 4, 16),
        ("hybrid1", 2, 4, 32),
        #("data",   8, 4, 16)
    ]
    
    # Setup the localized PopTorch cache directory
    CACHE_DIR = "./poptorch_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print(f"\n{('*'*90)}")
    print(f"🛠️  Orchestrating Sweep: {len(dataset_configs)} datasets, {len(loop_batch_sizes)} batch sizes, "
          f"{len(loop_model_names)} models, {len(loop_learn_rates)} learning rates, \n and {len(ipu_configs)} parallel modes")
    print('*'*90)
    

    try:
        # Combinatorial generation matching your exact loop architecture
        for batch_size, learn_rate, model_name, aug_dp in itertools.product(
            loop_batch_sizes, loop_learn_rates, loop_model_names, loop_aug_dps
        ):          
            for parallel_mode, rep_fact, device_itr, grad_accum in ipu_configs:
                for expr_type, data_path, data_name in dataset_configs:

                    # ONE-LINE CONTROL: Dynamically map required IPUs based on the parallel mode strategy
                    if parallel_mode == "model0":
                        ipu_ids = "0,1,2,3"
                    else:
                        ipu_ids = "0,1,2,3,4,5,6,7"

                    print(f"Experiment type in the wrapper= {expr_type}")

                    env = os.environ.copy()
                    env.update({
                        "POPTORCH_IPU_IDS": str(ipu_ids),
                        "POPTORCH_WAIT_FOR_IPU": "1",
                        #"RUN_EXPR_IDX": str(idx),
                        #"RUN_EXPR_TYPE": str(expr_type),
                        "RUN_BATCH_SIZE": str(batch_size),
                        "RUN_LEARN_RATE": str(learn_rate),
                        "RUN_MODEL_NAME": str(model_name),
                        "RUN_PARALLEL_MODE": str(parallel_mode),
                        "RUN_AUG_DP": str(aug_dp),
                        "RUN_REP_FACT": str(rep_fact),
                        "RUN_DEVICE_ITR": str(device_itr),
                        "RUN_GRAD_ACCUM": str(grad_accum),
                        "RUN_EXPR_TYPE": str(expr_type),
                        "RUN_DATA_PATH": str(data_path),
                        "RUN_DATA_NAME": str(data_name),

                    })

                    proc = None

                    try:
                        with open("sweep_error.log", "w") as err_log:
                            proc = subprocess.Popen(
                                ["python", "Iputrain-main.py"], env=env,
                                stderr=subprocess.DEVNULL  # suppress the Traceback notices

                            )
                            returncode = proc.wait()

                        if returncode != 0:
                            raise subprocess.CalledProcessError(returncode, proc.args)

                    except KeyboardInterrupt:
                        print("\n🛑 Interrupt received. Waiting for training shutdown...")

                        if proc is not None and proc.poll() is None:
                            try:
                                proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait()

                        raise SystemExit(130)
                    
    finally:
        # AUTO-CLEAR DISK SPACE: Cleans up the huge binaries after the loops complete or if interrupted
        print(f"\n🧹 Cleaning up intermediate PopTorch cache to save disk space...")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        print("✅ Cache cleared. Only your final trained .pt models and benchmark CSVs remain!")

if __name__ == "__main__":
    main()
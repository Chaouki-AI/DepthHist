#!/bin/bash

# Load the Anaconda environment configuration
echo "Sourcing Anaconda configuration..."
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the DepthHist environment
echo "Activating conda environment: DepthHist"
conda activate DepthHist

# ----------------------------------------
# Run training and evaluation on KITTI
# ----------------------------------------

echo "Running training on KITTI dataset..."
python main.py @args/train/args_kitti.txt

echo "Evaluating on KITTI dataset..."
python evaluator.py @args/evals/args_kitti.txt

# ----------------------------------------
# Run training and evaluation on NYUv2
# ----------------------------------------

echo "Running training on NYUv2 dataset..."
python main.py @args/train/args_nyu.txt

echo "Evaluating on NYUv2 dataset..."
python evaluator.py @args/evals/args_nyu.txt

# Deactivate the conda environment
echo "Deactivating conda environment: DepthHist"
conda deactivate


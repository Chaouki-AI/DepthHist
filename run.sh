#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate DepthHist


# Run the first command
#python main.py @args/args_kitti_bins_cauchy.txt
python main.py @args/args_nyu_bins_cauchy.txt

# Deactivate the run
conda deactivate
#!/bin/bash
#SBATCH --job-name=Swin-UNETR_postprocessing

#SBATCH --partition=gpu

#SBATCH --cpus-per-task 10

#SBATCH --mem 50G

#SBATCH --output=/experiments_Swin-UNETR/post-processing/results/postprocessing.log

#SBATCH --gres=gpu:1

module load PyTorch

export nnUNet_raw="/nnUNet/raw_data"
export nnUNet_preprocessed="/nnUNet/preprocessed_data"
export nnUNet_results="/nnUNet/nnUNet_results"

export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
export PATH="/.local/bin:/usr/local/cuda-11.7/bin:$PATH"
export PYTHONPATH="/.local/lib/python3.10/site-packages:$PYTHONPATH:"


python experiments_Swin-UNETR/post-processing/postprocessing_multiprocess.py --pp_config 1 --rate 0.1


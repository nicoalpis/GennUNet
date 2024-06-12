#!/bin/bash
#SBATCH --job-name=nnUNet_postprocessing

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 1

#SBATCH --mem 10G

#SBATCH --output=/nnUNet_inference/post-processing/results/postprocessing.out

#SBATCH --gres=gpu:0

module load PyTorch

export nnUNet_raw="/nnUNet/raw_data"
export nnUNet_preprocessed="/nnUNet/preprocessed_data"
export nnUNet_results="/nnUNet/nnUNet_results"

export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
export PATH="/.local/bin:/usr/local/cuda-11.7/bin:$PATH"
export PYTHONPATH="/.local/lib/python3.10/site-packages:$PYTHONPATH:"


python nnUNet_inference/postprocessing_multiprocess.py --pp_config 0


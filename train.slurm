#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1  # 32 processor core(s) per node 
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=priority-a100    # gpu node(s)
#SBATCH -A baskargroup-a100gpu
#SBATCH --no-requeue
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load miniconda3/4.10.3-svrr7oe
source activate DLRG_Lightning

###### Masked AutoEncoder ######
python main.py -d -g 2 --num_workers 1 -ep 100 --proj_name Masked_AutoEncoder --name MAE_WithoutBackground_bs128_ep100 -ae -b 128

#python main.py -g 4 --num_workers 1 -ep 100 --proj_name AutoEncoder --name AE_WithoutBackground_bs128_ep100_GroupNorm -ae -b 128 --gn --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithoutBackground_bs128_ep100_GroupNorm/version_5/last.ckpt'

#python main.py -g 4 --num_workers 1 -ep 100 --proj_name AutoEncoder --name AE_WithBackground_bs128_ep100_GroupNorm -ae -bg -b 128 --gn

#python main.py -g 4 --num_workers 1 -ep 100 --proj_name AutoEncoder --name AE_WithBackground_bs128_ep100_GroupNorm -ae -bg -b 128 --gn --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100_GroupNorm/version_7/last.ckpt'

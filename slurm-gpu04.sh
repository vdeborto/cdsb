#!/bin/bash

# Usage: sbatch slurm.sh
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out
#SBATCH --job-name=cdsb_develop
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
# only increase it when explicitly setting --nodelist=zizgpu04.cpu.stats.ox.ac.uk (see below)
# on zizgpu04 use --cpus-per-task=3, max --cpus-per-task=5 if your job really needs it

#SBATCH --nodelist=zizgpu04.cpu.stats.ox.ac.uk
# You can enable this by changing NOTSBATCH to SBATCH.
# This way you can request for your jobs to be run on a particular node (mostly useful to select zizgpu04 because of larger CPU capacity, so you can set --cpus-per-gpu>1).
# If you provide more than one node in the --nodelist it will have unintuitive consequences.

#SBATCH --time=14-00:00:00  # kills job if it runs for longer than this time, 14 days is the maximum
#SBATCH --mem=160G
#SBATCH --ntasks=1

export PATH_TO_CONDA="/data/localhost/not-backed-up/yshi/miniconda3"

# Activate conda virtual environment
source ~/.bashrc
source $PATH_TO_CONDA/bin/activate bridge

{ echo 0; echo 2; echo 1; echo NO; echo 4; echo no; } | accelerate config
accelerate launch main.py dataset=celeba cond_final=True final_adaptive=True gamma_max=0.01 num_iter=1000000 model.use_fp16=True


echo "Job completed."

rsync /tmp/slurm-${SLURM_JOB_ID}.out /data/ziz/not-backed-up/scratch/yshi/slurm/cdsb_develop/
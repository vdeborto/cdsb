#!/bin/bash

# Usage: sbatch slurm.sh
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out
#SBATCH --job-name=cdsb_develop
#SBATCH --partition=ziz-gpu-titan
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
# only increase it when explicitly setting --nodelist=zizgpu04.cpu.stats.ox.ac.uk (see below)
# on zizgpu04 use --cpus-per-task=3, max --cpus-per-task=5 if your job really needs it

#SBATCH --nodelist=zizgpu05.cpu.stats.ox.ac.uk
# You can enable this by changing NOTSBATCH to SBATCH.
# This way you can request for your jobs to be run on a particular node (mostly useful to select zizgpu04 because of larger CPU capacity, so you can set --cpus-per-gpu>1).
# If you provide more than one node in the --nodelist it will have unintuitive consequences.

#SBATCH --time=14-00:00:00  # kills job if it runs for longer than this time, 14 days is the maximum
#SBATCH --mem=160G  # RAM
#SBATCH --ntasks=1

export PATH_TO_CONDA="/data/localhost/not-backed-up/yshi/miniconda3"

# Activate conda virtual environment
source ~/.bashrc
source $PATH_TO_CONDA/bin/activate bridge1

# python main.py num_iter=200000 gamma_max=0.02 cond_final=True final_adaptive=True mean_match=False # can use hydra for this

{ echo 0; echo 2; echo 1; echo NO; echo 2; echo no; } | accelerate config
# { echo 0; echo 0; echo NO; echo 1; echo no; } | accelerate config
# accelerate launch main.py num_iter=200000 cond_final=True final_adaptive=True gamma_max=0.03
# accelerate launch --main_process_port 29501 main.py num_iter=50000 cond_final=True final_adaptive=True gamma_max=0.03
# accelerate launch --main_process_port 29504 main.py num_iter=500000 cache_refresh_stride=500

# cond_final=True final_adaptive=True gamma_max=0.01 num_steps=50 batch_size=256


# accelerate launch mnist.py dataset=celeba cond_final=True final_adaptive=True gamma_max=0.01 num_iter=50000 symmetric_gamma=False model.use_fp16=True
# accelerate launch mnist.py dataset=celeba cond_final=True final_adaptive=True gamma_max=0.01 num_iter=100000 model.use_fp16=True
accelerate launch mnist.py dataset=celeba num_iter=20000 n_ipf=25

echo "Job completed."

rsync /tmp/slurm-${SLURM_JOB_ID}.out /data/ziz/not-backed-up/scratch/yshi/slurm/cdsb_develop/
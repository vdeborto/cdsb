#!/bin/bash

# Usage: sbatch slurm.sh
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out
#SBATCH --job-name=cdsb_develop
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
# only increase it when explicitly setting --nodelist=zizgpu04.cpu.stats.ox.ac.uk (see below)
# on zizgpu04 use --cpus-per-task=3, max --cpus-per-task=5 if your job really needs it

#SBATCH --nodelist=zizgpu04.cpu.stats.ox.ac.uk
# You can enable this by changing NOTSBATCH to SBATCH.
# This way you can request for your jobs to be run on a particular node (mostly useful to select zizgpu04 because of larger CPU capacity, so you can set --cpus-per-gpu>1).
# If you provide more than one node in the --nodelist it will have unintuitive consequences.

#SBATCH --time=14-00:00:00  # kills job if it runs for longer than this time, 14 days is the maximum
#SBATCH --mem=20G
#SBATCH --ntasks=1

export PATH_TO_CONDA="/data/localhost/not-backed-up/yshi/miniconda3"

# Activate conda virtual environment
source ~/.bashrc
source $PATH_TO_CONDA/bin/activate bridge

# { echo 0; echo 2; echo 1; echo NO; echo 2; echo no; } | accelerate config
# { echo 0; echo 2; echo 1; echo NO; echo 4; echo no; } | accelerate config

# python mnist.py num_iter=50000 n_ipf=10 num_steps=5 num_cache_batches=8 model.dropout=0.1
python mnist.py num_iter=100000 n_ipf=5 num_steps=10 num_cache_batches=4 model.dropout=0.05
# python mnist.py cond_final=True final_adaptive=True gamma_max=0.03 num_iter=20000 n_ipf=25 num_steps=10 num_cache_batches=4 model.dropout=0.05
# accelerate launch main.py num_iter=1000000 n_ipf=1 num_steps=10 num_cache_batches=4 model.dropout=0.05
# accelerate launch main.py cond_final=True final_adaptive=True gamma_max=0.03 num_iter=10000000 n_ipf=1
# accelerate launch --main_process_port 29502 main.py num_iter=10000 cond_final=True final_adaptive=True num_steps=10 num_cache_batches=4 gamma_max=0.05
# accelerate launch --main_process_port 29504 main.py num_iter=10000000 n_ipf=1 dataset=mnist_inpaint model.dropout=0.05
# accelerate launch --main_process_port 29508 main.py num_iter=100000 n_ipf=10 cond_final=True final_adaptive=True gamma_max=0.05 num_steps=10 num_cache_batches=4 model.dropout=0.05


# cond_final=True final_adaptive=True gamma_max=0.01 num_steps=50 batch_size=256


echo "Job completed."

rsync /tmp/slurm-${SLURM_JOB_ID}.out /data/ziz/not-backed-up/scratch/yshi/slurm/cdsb_develop/
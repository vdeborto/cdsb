#!/bin/bash

# Usage: sbatch slurm.sh
#SBATCH --output=./slurm/slurm-%j.out
#SBATCH --error=./slurm/slurm-%j.out
#SBATCH --job-name=cdsb_develop
#SBATCH --partition=short
#SBATCH --gres=gpu:a6000:2  # --gres=gpu:2  # --gres=gpu:rtx:2  # --gres=gpu:v100:2  # --gres=gpu:a100:2  #--gres=gpu:rtx8000:1  #  
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00  # kills job if it runs for longer than this time
#SBATCH --mem=100G  # RAM
#SBATCH --ntasks=1
#NOTSBATCH --nodelist=htc-g044
#NOTSBATCH --exclude=htc-g044,htc-g045,htc-g048
#SBATCH --exclude=htc-g[044-049]

nvidia-smi

export CONPREFIX=$DATA/bridge

# Activate conda virtual environment
module load Anaconda3/2021.11
source activate $CONPREFIX

# CD to TMPDIR / SCRATCH
# cd $SCRATCH || exit 1

# rsync -avz --include='*.py' --exclude='*' $DATA/cdsb_develop/ ./
# rsync -avz $DATA/cdsb_develop/bridge $DATA/cdsb_develop/conf ./
# rsync -az --exclude='mnist/' $DATA/cdsb_develop/data ./

{ echo 0; echo 2; echo 1; echo NO; echo 2; echo no; } | accelerate config
# { echo 0; echo 2; echo 1; echo NO; echo 4; echo no; } | accelerate config
# { echo 0; echo 0; echo NO; echo 1; echo no; } | accelerate config

accelerate launch mnist.py dataset=celeba_inpaint num_iter=500000 n_ipf=1 gamma_max=0.05 num_cache_batches=15
# accelerate launch --main_process_port 29501 mnist.py dataset=celeba_inpaint num_iter=500000 n_ipf=1 num_steps=20 num_cache_batches=35 

# accelerate launch regression.py dataset=celeba_inpaint num_iter=100000 gamma_max=0.005 plot_npar=2000 

# rsync -auvz ./experiments/ $DATA/cdsb_develop/experiments/

echo "Job completed."

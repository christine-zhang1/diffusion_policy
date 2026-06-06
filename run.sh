#!/bin/bash
#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --exclude=jagupard[19-36]
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=jag-standard
#SBATCH --time=24:00:00
#SBATCH --job-name="diffusion_policy"
#SBATCH --output=/nlp/scr/chrzhang/logs/diffusion_policy-%j.out
#SBATCH --error=/nlp/scr/chrzhang/logs/diffusion_policy-%j.err

# SBATCH --account=move
# SBATCH --partition=move --qos=normal
# SBATCH --time=24:00:00
# SBATCH --nodes=1
# SBATCH --cpus-per-task=4
# SBATCH --mem-per-cpu=32G

# only use the following on partition with GPUs
# SBATCH --gres=gpu:titanrtx:1

# SBATCH --job-name="diffusion_policy"
# SBATCH --output=logs/diffusion_policy-%j.out
# SBATCH --error=logs/diffusion_policy-%j.err

# only use the following if you want email notification
# SBATCH --mail-user=chrzhang@stanford.edu
# SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# not needed if already in the conda environment when running this script
source /nlp/scr/chrzhang/miniconda3/etc/profile.d/conda.sh
conda activate dp
# HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# python train.py --config-name=train_diffusion_unet_simtool_workspace
python train.py --config-name=train_diffusion_transformer_simtool_workspace

# done
echo "Done"

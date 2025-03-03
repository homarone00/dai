#!/bin/bash
#SBATCH -e /homes/ocarpentiero/output/err.txt
#SBATCH -o /homes/ocarpentiero/output/out.txt
#SBATCH --job-name=simclr50_train_norm
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=24:00:00

source /homes/ocarpentiero/dai/.venv/bin/activate
python /homes/ocarpentiero/dai/multiwalker_v1/main.py
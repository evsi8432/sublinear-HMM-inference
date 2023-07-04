#!/bin/bash
#SBATCH --account=def-nheckman

#SBATCH --mail-user=evan.sidrow@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-400

source ~/Whale_Paper_1/bin/activate
module load python/3.9
module load scipy-stack

python sim_study.py 12 $SLURM_ARRAY_TASK_ID

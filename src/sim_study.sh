#!/bin/bash
#SBATCH --account=def-nheckman

#SBATCH --mail-user=evan.sidrow@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-100

if [ $SLURM_ARRAY_TASK_ID -lt 50 ]
then
  python sim_study.py 1 10000 10 $SLURM_ARRAY_TASK_ID
fi
if [ $SLURM_ARRAY_TASK_ID -gt 49 ]
then
  python sim_study.py 2 10000 10 $(($SLURM_ARRAY_TASK_ID - 50))
fi

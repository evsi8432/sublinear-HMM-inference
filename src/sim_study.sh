#!/bin/bash
#SBATCH --account=def-nheckman

#SBATCH --mail-user=evan.sidrow@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-200

if [ $SLURM_ARRAY_TASK_ID -lt 101 ]
then
  python sim_study.py 1 $SLURM_ARRAY_TASK_ID
fi
if [ $SLURM_ARRAY_TASK_ID -gt 100 ]
then
  python sim_study.py 2 $SLURM_ARRAY_TASK_ID 
fi

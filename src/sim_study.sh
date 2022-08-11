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
#SBATCH --array=1-273

python sim_study.py 3 10000 10 $SLURM_ARRAY_TASK_ID

# get the control studies
if [ $SLURM_ARRAY_TASK_ID == 271 ]
then
  python sim_study.py 1 10000 10 -1
fi
if [ $SLURM_ARRAY_TASK_ID == 272 ]
then
  python sim_study.py 2 10000 10 -1
fi
if [ $SLURM_ARRAY_TASK_ID == 273 ]
then
  python sim_study.py 3 10000 10 -1
fi

# get everything else
if [ $SLURM_ARRAY_TASK_ID -gt 0 ] && [ $SLURM_ARRAY_TASK_ID -lt 90 ]
then
  python sim_study.py 1 10000 10 $SLURM_ARRAY_TASK_ID
fi
if [ $SLURM_ARRAY_TASK_ID -gt 89 ] && [ $SLURM_ARRAY_TASK_ID -lt 180 ]
then
  python sim_study.py 2 10000 10 $(($SLURM_ARRAY_TASK_ID - 90))
fi
if [ $SLURM_ARRAY_TASK_ID -gt 179 ] && [ $SLURM_ARRAY_TASK_ID -lt 270 ]
then
  python sim_study.py 3 10000 10 $(($SLURM_ARRAY_TASK_ID - 180))
fi

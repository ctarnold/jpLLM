#!/bin/bash
#SBATCH --job-name=mistralTest2         # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=180G        # memory per cpu-core (4G is default)

#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ca2992@princeton.edu

module purge
module load anaconda3/2024.2
module load cudatoolkit/12.2
conda activate jpllm

python mistalPrompting.py
python mistralPromptingMix.py
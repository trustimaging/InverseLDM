#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=presbrainaxial
#SBATCH --partition=brain_main
#SBATCH --qos=normal

# Each job will utilise all of brainstorm's resources.
# Here, we ask for 1 node with exclusive access.
# brainstorm has 256 CPU cores and 8 A100 GPUs.

#SBATCH -e /raid/dverschu/lfr/slurm_outputs/slurm-%j.err              # File to redirect stderr
#SBATCH -o /raid/dverschu/lfr/slurm_outputs/slurm-%j.out              # File to redirect stdout
#SBATCH --mem=1900gb              # Total memory for the job (1500 MB * 256 CPUs)
#SBATCH --time=20:00:00              # The walltime
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=2                   # Number of tasks
##SBATCH --ntasks-per-socket=1       # Maximum number of tasks on each socket
#SBATCH --cpus-per-task=128          # Total CPUs for the task
#SBATCH --gres=gpu:8                 # Number of GPUs
# This is where the actual work is done.



# activate conda environment
source /raid/dverschu/miniconda3/etc/profile.d/conda.sh
conda activate diff2

python /raid/dverschu/InverseLDM/train.py --config /raid/dverschu/InverseLDM/config.yml --name strong_conditioning_new --overwrite -y --gpu_ids [0,1,2,3,4,5,6,7] 
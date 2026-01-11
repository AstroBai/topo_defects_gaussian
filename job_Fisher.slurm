#!/bin/tcsh
#SBATCH --time=48:00:00                     # set max wallclock time
#SBATCH --output=job-%N-%j.out             # set output log name
#SBATCH --error=job-%N-%j.error            # set error log name %N=first node, %j=job number
#SBATCH --job-name=FisherPipeline_gaussian         # set job name
#SBATCH -p all                             # set partition (queue)
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --mail-type=ALL                    # set mail alerts
#SBATCH --mail-user=j.bai@ed.ac.uk # set mail address

setenv OMP_NUM_THREADS $SLURM_JOB_CPUS_PER_NODE
./FisherPipeline.sh

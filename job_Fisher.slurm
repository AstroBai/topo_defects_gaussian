#!/bin/bash
#SBATCH --time=0:30:00                     # set max wallclock time
#SBATCH --output=job-%N-%j.out             # set output log name
#SBATCH --error=job-%N-%j.error            # set error log name %N=first node, %j=job number
#SBATCH --job-name=FisherPipeline_gaussian         # set job name
#SBATCH -p all                             # set partition (queue)
#SBATCH --nodes=1                          # set node number
#SBATCH --ntasks-per-node=1                # set task number
#SBATCH --mem-per-cpu=4GB                  # set memory per core
#SBATCH --mail-type=ALL                    # set mail alerts
# Uncomment line below to get more emails.
#SBATCH --mail-user=j.bai@ed.ac.uk # set mail address

./FisherPipeline.sh

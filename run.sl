#!/bin/bash
#
#SBATCH --job-name=tf_idf
#SBATCH --output=tf_idf.txt
#
##SBATCH --cpus-per-task=10
#number of CPUs to be used
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
##SBATCH --nodes=1
##SBATCH --ntasks-per-core=1
##SBATCH --ntasks-per-core=100
##SBATCH --overcommit
#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=24:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=64G

##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
##SBATCH --constraint=A10
##SBATCH --constraint=L40S
##SBATCH --nodelist=gpu227
##SBATCH --constraint=zeta
#SBATCH --constraint=eta

#Send emails when a job starts, it is finished or it exits
#SBATCH --mail-user=matinansaripour@gmail.com
#SBATCH --mail-type=END,FAIL
#
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV


#cd $HOME/Jupyter/Hybrid-Decentralized-Optimization
#source ./venv/bin/activate

source $HOME/.bashrc
conda activate DIS_project1

#module load openmpi/4.1.6
#module load cuda/12
#module load cudnn/8.9.5.30

#export HWLOC_COMPONENTS=-gl



python3 tfidf/indexing.py -dir /nfs/scistore16/krishgrp/mansarip/Jupyter/DIS_project1/data -lang "en"
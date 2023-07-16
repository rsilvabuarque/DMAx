#!/bin/bash
#Job name:
#SBATCH --job-name=structure_name_charge_analysis
#
#Partition:
#SBATCH --partition=shared
#
#Account
#SBATCH --account=csd626
#
#Nodes
#SBATCH --nodes=1
#
#Processors:
#SBATCH --ntasks-per-node=32
#
#Wall clock limit:
#SBATCH --time=24:00:00
#
#SBATCH --mail-type=ALL
#

module purge
module load cpu/0.15.4  gcc/10.2.0  mvapich2/2.3.6
module load qchem/6.0.2

qchem structure_name.in > structure_name.out

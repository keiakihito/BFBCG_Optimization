#!/bin/bash
# Check if SLURM_PROCID is set
if [ -z "$SLURM_PROCID" ]; then
  echo "SLURM_PROCID is not set. Exiting..."
  exit 1
fi

# Use $PMI_RANK for MPICH, $OMPI_COMM_WORLD_RANK for openmpi, and $SLURM_PROCID with srun.
if [ $SLURM_PROCID -eq 0 ]; then
  # Run profiling with nsys for the master process
  nsys profile -o ${OUTPUT_DIR}/mynsys.out --stats=true "$@"
else
  # Run the command as normal for other processes
  "$@"
fi


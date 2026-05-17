#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N hybrid_dynamic_conv

# Example SGE script. Adjust the parallel environment name and core count to Moore's configuration.
# Example: 4 MPI processes x 4 OpenMP threads/process = 16 total cores.

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
make
mpirun -np ${NSLOTS:-4} ./hybridconv \
  /share/apps/files/convolution/images/im03.ppm \
  /share/apps/files/convolution/kernel/Kernel49x49_Random2.txt \
  out_im03_k49_hybrid_dynamic.ppm \
  1 dynamic 128

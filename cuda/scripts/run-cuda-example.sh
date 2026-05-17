#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N cuda_conv

# Adjust this script to the CUDA server or queue environment used by the course.
# Do not execute long runs on a login/front-end node.

make
./cudaconv \
  /share/apps/files/convolution/images/im03.ppm \
  /share/apps/files/convolution/kernel/Kernel49x49_Random2.txt \
  out_im03_k49_cuda_16x16.ppm \
  1 16 16

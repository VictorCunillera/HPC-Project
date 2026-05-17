#!/bin/bash

make
./cudaconv \
  /share/apps/files/convolution/images/im03.ppm \
  /share/apps/files/convolution/kernel/Kernel49x49_Random2.txt \
  out_im03_k49_cuda_16x16.ppm \
  1 16 16

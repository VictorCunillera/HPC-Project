# HPC Project - Convolution Operation

This repository contains the source code for the last two project deliveries:

- `hybrid_mpi_openmp/`: Delivery 2, Hybrid MPI + OpenMP implementation.
- `cuda/`: Delivery 3, CUDA implementation.

Both implementations keep the same program interface used in the assignment:

```bash
program image.ppm kernel.txt output.ppm partitions [extra options]
```

The input images and kernels should be read from the shared folders in the cluster/server:

```text
/share/apps/files/convolution/images
/share/apps/files/convolution/kernel
```

Do not copy the original images into `$HOME`, and do not execute long jobs on the front-end node. Use the corresponding queue scripts or the execution mechanism provided by the subject.

## Delivery 2 - Hybrid MPI + OpenMP

Build:

```bash
cd hybrid_mpi_openmp
make
```

Static mapping example:

```bash
export OMP_NUM_THREADS=4
mpirun -np 4 ./hybridconv \
  /share/apps/files/convolution/images/im03.ppm \
  /share/apps/files/convolution/kernel/Kernel49x49_Random2.txt \
  out_im03_k49_hybrid_static.ppm \
  1 static
```

Dynamic mapping example:

```bash
export OMP_NUM_THREADS=4
mpirun -np 4 ./hybridconv \
  /share/apps/files/convolution/images/im03.ppm \
  /share/apps/files/convolution/kernel/Kernel49x49_Random2.txt \
  out_im03_k49_hybrid_dynamic.ppm \
  1 dynamic 128
```

The final optional argument in dynamic mode is the number of rows per task chunk.

## Delivery 3 - CUDA

Build:

```bash
cd cuda
make
```

CUDA execution example:

```bash
./cudaconv \
  /share/apps/files/convolution/images/im03.ppm \
  /share/apps/files/convolution/kernel/Kernel49x49_Random2.txt \
  out_im03_k49_cuda.ppm \
  1 16 16
```

The optional last two arguments are the CUDA block dimensions, for example `8 8`, `16 16`, or `32 8`.

## Timing

- The Hybrid version uses `MPI_Wtime()` for distributed execution timing.
- The CUDA version uses CUDA events for GPU kernel timing and CPU wall-clock timing for total elapsed time.

## Notes

The output image is written as P6 PPM. Resulting images can be large; transfer them to your local machine if needed and remove them from the cluster/server afterwards.

/*
 * HPC Project - Convolution Operation
 * Delivery 2: Hybrid MPI + OpenMP implementation
 *
 * Usage:
 *   mpirun -np <processes> ./hybridconv image.ppm kernel.txt output.ppm partitions [static|dynamic] [chunk_rows]
 *
 * Design summary:
 *   - MPI decomposes the image by horizontal row blocks/chunks.
 *   - OpenMP parallelizes the pixel loop inside each MPI process.
 *   - Static mode assigns one fixed row interval to each process.
 *   - Dynamic mode uses rank 0 as master and the remaining ranks as workers.
 *   - MPI_Wtime() is used for timing.
 *
 * Notes:
 *   - The input image is broadcast to all MPI ranks. This keeps the convolution
 *     kernel simple and avoids halo-exchange errors, because every rank can read
 *     neighbour pixels safely. The output is gathered on rank 0.
 *   - P6 and P3 PPM images are supported. Output is written as P6.
 */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define TAG_REQUEST 10
#define TAG_TASK    11
#define TAG_RESULT  12
#define TAG_DATA    13
#define TAG_STOP    14

#define MAX_KERNEL_VALUES 10000

typedef struct {
    int width;
    int height;
    int maxval;
    unsigned char *rgb;
} Image;

typedef struct {
    int size;
    float *values;
} Kernel;

static void die_root(int rank, const char *msg) {
    if (rank == 0) fprintf(stderr, "%s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

static void skip_ws_and_comments(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n') {}
            continue;
        }
        ungetc(c, f);
        return;
    }
}

static int read_token(FILE *f, char *buf, size_t n) {
    skip_ws_and_comments(f);
    size_t i = 0;
    int c;
    while ((c = fgetc(f)) != EOF && !isspace(c)) {
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n') {}
            break;
        }
        if (i + 1 < n) buf[i++] = (char)c;
    }
    buf[i] = '\0';
    return i > 0;
}

static Image read_ppm(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(EXIT_FAILURE); }

    char tok[256];
    if (!read_token(f, tok, sizeof(tok))) { fprintf(stderr, "Invalid PPM file\n"); exit(EXIT_FAILURE); }

    int binary = 0;
    if (strcmp(tok, "P6") == 0) binary = 1;
    else if (strcmp(tok, "P3") == 0) binary = 0;
    else { fprintf(stderr, "Unsupported PPM format: %s\n", tok); exit(EXIT_FAILURE); }

    if (!read_token(f, tok, sizeof(tok))) { fprintf(stderr, "Missing width\n"); exit(EXIT_FAILURE); }
    int w = atoi(tok);
    if (!read_token(f, tok, sizeof(tok))) { fprintf(stderr, "Missing height\n"); exit(EXIT_FAILURE); }
    int h = atoi(tok);
    if (!read_token(f, tok, sizeof(tok))) { fprintf(stderr, "Missing maxval\n"); exit(EXIT_FAILURE); }
    int maxv = atoi(tok);

    if (w <= 0 || h <= 0 || maxv <= 0 || maxv > 255) {
        fprintf(stderr, "Invalid PPM header\n");
        exit(EXIT_FAILURE);
    }

    size_t n = (size_t)w * (size_t)h * 3u;
    unsigned char *data = (unsigned char*)malloc(n);
    if (!data) { fprintf(stderr, "Not enough memory for image\n"); exit(EXIT_FAILURE); }

    if (binary) {
        int c = fgetc(f);
        if (c != EOF && !isspace(c)) ungetc(c, f);
        if (fread(data, 1, n, f) != n) {
            fprintf(stderr, "Could not read P6 image data\n");
            exit(EXIT_FAILURE);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            if (!read_token(f, tok, sizeof(tok))) {
                fprintf(stderr, "Could not read P3 pixel data\n");
                exit(EXIT_FAILURE);
            }
            int v = atoi(tok);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            data[i] = (unsigned char)v;
        }
    }

    fclose(f);
    Image img = {w, h, maxv, data};
    return img;
}

static void write_ppm(const char *path, const Image *img) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(EXIT_FAILURE); }
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    size_t n = (size_t)img->width * (size_t)img->height * 3u;
    if (fwrite(img->rgb, 1, n, f) != n) {
        fprintf(stderr, "Could not write output image\n");
        exit(EXIT_FAILURE);
    }
    fclose(f);
}

static Kernel read_kernel(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(EXIT_FAILURE); }

    size_t cap = 1024, count = 0;
    float *vals = (float*)malloc(cap * sizeof(float));
    if (!vals) { fprintf(stderr, "Out of memory\n"); exit(EXIT_FAILURE); }

    while (1) {
        float x;
        int r = fscanf(f, "%f", &x);
        if (r == 1) {
            if (count == cap) {
                cap *= 2;
                float *tmp = (float*)realloc(vals, cap * sizeof(float));
                if (!tmp) { fprintf(stderr, "Out of memory\n"); exit(EXIT_FAILURE); }
                vals = tmp;
            }
            vals[count++] = x;
        } else if (r == EOF) {
            break;
        } else {
            fgetc(f);
        }
    }
    fclose(f);

    int size = 0;
    float *k = NULL;

    if (count >= 3) {
        int a = (int)vals[0];
        int b = (int)vals[1];
        if ((float)a == vals[0] && (float)b == vals[1] && a == b && a > 0 && (size_t)(a * b + 2) == count) {
            size = a;
            k = (float*)malloc((size_t)size * (size_t)size * sizeof(float));
            if (!k) { fprintf(stderr, "Out of memory\n"); exit(EXIT_FAILURE); }
            memcpy(k, vals + 2, (size_t)size * (size_t)size * sizeof(float));
        }
    }

    if (!k) {
        int s = (int)(sqrt((double)count) + 0.5);
        if (s <= 0 || (size_t)(s * s) != count || s * s > MAX_KERNEL_VALUES) {
            fprintf(stderr, "Kernel file does not contain a valid square matrix\n");
            exit(EXIT_FAILURE);
        }
        size = s;
        k = (float*)malloc((size_t)size * (size_t)size * sizeof(float));
        if (!k) { fprintf(stderr, "Out of memory\n"); exit(EXIT_FAILURE); }
        memcpy(k, vals, (size_t)size * (size_t)size * sizeof(float));
    }

    free(vals);
    Kernel ker = {size, k};
    return ker;
}

static unsigned char clamp_byte(float x) {
    int v = (int)lrintf(x);
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (unsigned char)v;
}

static void convolve_rows_omp(const unsigned char *in, unsigned char *out,
                              int width, int height,
                              const float *kernel, int ksize,
                              int row_start, int row_end) {
    int radius = ksize / 2;

    #pragma omp parallel for schedule(static)
    for (int y = row_start; y < row_end; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float acc = 0.0f;
                for (int ky = 0; ky < ksize; ++ky) {
                    int yy = y + ky - radius;
                    if (yy < 0 || yy >= height) continue;
                    for (int kx = 0; kx < ksize; ++kx) {
                        int xx = x + kx - radius;
                        if (xx < 0 || xx >= width) continue;
                        acc += (float)in[((yy * width + xx) * 3) + c] * kernel[ky * ksize + kx];
                    }
                }
                out[(((y - row_start) * width + x) * 3) + c] = clamp_byte(acc);
            }
        }
    }
}

static void broadcast_input(int rank, Image *img, Kernel *ker) {
    int meta[4];
    if (rank == 0) {
        meta[0] = img->width;
        meta[1] = img->height;
        meta[2] = img->maxval;
        meta[3] = ker->size;
    }
    MPI_Bcast(meta, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        img->width = meta[0];
        img->height = meta[1];
        img->maxval = meta[2];
        ker->size = meta[3];
        img->rgb = NULL;
        ker->values = NULL;
    }

    if (ker->size <= 0 || ker->size * ker->size > MAX_KERNEL_VALUES) {
        die_root(rank, "Invalid kernel size");
    }

    size_t image_bytes = (size_t)img->width * (size_t)img->height * 3u;
    size_t kernel_count = (size_t)ker->size * (size_t)ker->size;

    if (rank != 0) {
        img->rgb = (unsigned char*)malloc(image_bytes);
        ker->values = (float*)malloc(kernel_count * sizeof(float));
        if (!img->rgb || !ker->values) die_root(rank, "Out of memory after broadcast metadata");
    }

    MPI_Bcast(img->rgb, (int)image_bytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(ker->values, (int)kernel_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

static void static_mapping(int rank, int nproc, const Image *img, const Kernel *ker,
                           unsigned char **global_output, double *local_compute_time) {
    int base = img->height / nproc;
    int rem = img->height % nproc;
    int rows = base + (rank < rem ? 1 : 0);
    int start = rank * base + (rank < rem ? rank : rem);
    int end = start + rows;

    size_t local_bytes = (size_t)rows * (size_t)img->width * 3u;
    unsigned char *local_out = (unsigned char*)malloc(local_bytes > 0 ? local_bytes : 1);
    if (!local_out) die_root(rank, "Out of memory for local static output");

    double t0 = MPI_Wtime();
    convolve_rows_omp(img->rgb, local_out, img->width, img->height, ker->values, ker->size, start, end);
    double t1 = MPI_Wtime();
    *local_compute_time = t1 - t0;

    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        *global_output = (unsigned char*)malloc((size_t)img->width * (size_t)img->height * 3u);
        recvcounts = (int*)malloc((size_t)nproc * sizeof(int));
        displs = (int*)malloc((size_t)nproc * sizeof(int));
        if (!*global_output || !recvcounts || !displs) die_root(rank, "Out of memory for static gather");
        int offset = 0;
        for (int r = 0; r < nproc; ++r) {
            int rrows = base + (r < rem ? 1 : 0);
            recvcounts[r] = rrows * img->width * 3;
            displs[r] = offset;
            offset += recvcounts[r];
        }
    }

    MPI_Gatherv(local_out, (int)local_bytes, MPI_UNSIGNED_CHAR,
                rank == 0 ? *global_output : NULL, recvcounts, displs,
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    free(local_out);
    free(recvcounts);
    free(displs);
}

static void dynamic_mapping(int rank, int nproc, const Image *img, const Kernel *ker,
                            unsigned char **global_output, int chunk_rows,
                            double *local_compute_time) {
    *local_compute_time = 0.0;

    if (nproc == 1) {
        size_t bytes = (size_t)img->width * (size_t)img->height * 3u;
        *global_output = (unsigned char*)malloc(bytes);
        if (!*global_output) die_root(rank, "Out of memory for dynamic serial fallback");
        double t0 = MPI_Wtime();
        convolve_rows_omp(img->rgb, *global_output, img->width, img->height, ker->values, ker->size, 0, img->height);
        *local_compute_time = MPI_Wtime() - t0;
        return;
    }

    if (rank == 0) {
        size_t total_bytes = (size_t)img->width * (size_t)img->height * 3u;
        *global_output = (unsigned char*)malloc(total_bytes);
        if (!*global_output) die_root(rank, "Out of memory for global dynamic output");

        int next_row = 0;
        int completed_rows = 0;
        int stopped_workers = 0;
        int workers = nproc - 1;

        while (stopped_workers < workers) {
            int header[2] = {0, 0};
            MPI_Status st;
            MPI_Recv(header, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            int src = st.MPI_SOURCE;

            if (st.MPI_TAG == TAG_REQUEST) {
                int task[2];
                if (next_row < img->height) {
                    task[0] = next_row;
                    task[1] = chunk_rows;
                    if (task[0] + task[1] > img->height) task[1] = img->height - task[0];
                    next_row += task[1];
                    MPI_Send(task, 2, MPI_INT, src, TAG_TASK, MPI_COMM_WORLD);
                } else {
                    task[0] = -1;
                    task[1] = 0;
                    MPI_Send(task, 2, MPI_INT, src, TAG_STOP, MPI_COMM_WORLD);
                    stopped_workers++;
                }
            } else if (st.MPI_TAG == TAG_RESULT) {
                int start = header[0];
                int rows = header[1];
                int count = rows * img->width * 3;
                MPI_Recv((*global_output) + ((size_t)start * img->width * 3u),
                         count, MPI_UNSIGNED_CHAR, src, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                completed_rows += rows;
                (void)completed_rows;
            }
        }
    } else {
        while (1) {
            int request[2] = {-1, 0};
            MPI_Send(request, 2, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);

            int task[2];
            MPI_Status st;
            MPI_Recv(task, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            if (st.MPI_TAG == TAG_STOP || task[0] < 0) break;

            int start = task[0];
            int rows = task[1];
            int end = start + rows;
            size_t local_bytes = (size_t)rows * (size_t)img->width * 3u;
            unsigned char *local_out = (unsigned char*)malloc(local_bytes > 0 ? local_bytes : 1);
            if (!local_out) die_root(rank, "Out of memory for local dynamic output");

            double t0 = MPI_Wtime();
            convolve_rows_omp(img->rgb, local_out, img->width, img->height, ker->values, ker->size, start, end);
            *local_compute_time += MPI_Wtime() - t0;

            int meta[2] = {start, rows};
            MPI_Send(meta, 2, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
            MPI_Send(local_out, (int)local_bytes, MPI_UNSIGNED_CHAR, 0, TAG_DATA, MPI_COMM_WORLD);
            free(local_out);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (argc < 6 || argc > 7) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s image.ppm kernel.txt output.ppm partitions [static|dynamic] [chunk_rows]\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *image_path = argv[1];
    const char *kernel_path = argv[2];
    const char *out_path = argv[3];
    int partitions = atoi(argv[4]);
    const char *mode = argv[5];
    int chunk_rows = (argc == 7) ? atoi(argv[6]) : 128;
    if (partitions <= 0) partitions = 1;
    if (chunk_rows <= 0) chunk_rows = 128;

    Image img = {0, 0, 255, NULL};
    Kernel ker = {0, NULL};

    double t_total_start = MPI_Wtime();
    if (rank == 0) {
        img = read_ppm(image_path);
        ker = read_kernel(kernel_path);
    }

    broadcast_input(rank, &img, &ker);
    MPI_Barrier(MPI_COMM_WORLD);

    unsigned char *global_output = NULL;
    double local_compute_time = 0.0;
    double t_parallel_start = MPI_Wtime();

    if (strcmp(mode, "static") == 0) {
        static_mapping(rank, nproc, &img, &ker, &global_output, &local_compute_time);
    } else if (strcmp(mode, "dynamic") == 0) {
        dynamic_mapping(rank, nproc, &img, &ker, &global_output, chunk_rows, &local_compute_time);
    } else {
        die_root(rank, "Mode must be 'static' or 'dynamic'");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_parallel_end = MPI_Wtime();

    double max_compute = 0.0, min_compute = 0.0, avg_compute = 0.0;
    MPI_Reduce(&local_compute_time, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute_time, &min_compute, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute_time, &avg_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        avg_compute /= (double)nproc;
        Image out = {img.width, img.height, 255, global_output};
        write_ppm(out_path, &out);
        double t_total_end = MPI_Wtime();
        double imbalance = (avg_compute > 0.0) ? ((max_compute - avg_compute) / avg_compute) * 100.0 : 0.0;

        printf("Image: %s\n", image_path);
        printf("Kernel: %s\n", kernel_path);
        printf("Output: %s\n", out_path);
        printf("Image size: %d x %d\n", img.width, img.height);
        printf("Kernel size: %d x %d\n", ker.size, ker.size);
        printf("Partitions argument: %d\n", partitions);
        printf("Mode: %s\n", mode);
        printf("MPI processes: %d\n", nproc);
        printf("OpenMP threads/process: %d\n", omp_get_max_threads());
        printf("Total cores: %d\n", nproc * omp_get_max_threads());
        printf("Parallel region time: %.6f seconds\n", t_parallel_end - t_parallel_start);
        printf("Total elapsed time: %.6f seconds\n", t_total_end - t_total_start);
        printf("Compute time max/min/avg: %.6f / %.6f / %.6f seconds\n", max_compute, min_compute, avg_compute);
        printf("Imbalance estimate: %.2f %%\n", imbalance);
    }

    free(img.rgb);
    free(ker.values);
    if (rank == 0) free(global_output);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

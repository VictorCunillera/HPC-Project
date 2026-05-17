/*
 * HPC Project - Convolution Operation
 * Delivery 3: CUDA implementation
 *
 * Usage:
 *   ./cudaconv image.ppm kernel.txt output.ppm partitions [block_x block_y]
 *
 * Design summary:
 *   - One CUDA thread computes one output pixel.
 *   - The kernel matrix is stored in CUDA constant memory.
 *   - R, G and B channels are computed inside the same CUDA thread to improve spatial locality.
 *   - CUDA events measure GPU kernel time, while CLOCK_MONOTONIC measures total application time.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#define MAX_KERNEL_VALUES 10000
__constant__ float cKernel[MAX_KERNEL_VALUES];

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

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

static double seconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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

    read_token(f, tok, sizeof(tok)); int w = atoi(tok);
    read_token(f, tok, sizeof(tok)); int h = atoi(tok);
    read_token(f, tok, sizeof(tok)); int maxv = atoi(tok);
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
            fprintf(stderr, "Could not read P6 pixels\n");
            exit(EXIT_FAILURE);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            if (!read_token(f, tok, sizeof(tok))) { fprintf(stderr, "Could not read P3 pixel\n"); exit(EXIT_FAILURE); }
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
    if (fwrite(img->rgb, 1, n, f) != n) { fprintf(stderr, "Could not write output image\n"); exit(EXIT_FAILURE); }
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

__global__ void convolution_kernel_rgb(const unsigned char *in, unsigned char *out,
                                       int width, int height, int ksize,
                                       int row_start, int row_end) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = row_start + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= row_end || y >= height) return;

    int radius = ksize / 2;
    for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        for (int ky = 0; ky < ksize; ++ky) {
            int yy = y + ky - radius;
            if (yy < 0 || yy >= height) continue;
            for (int kx = 0; kx < ksize; ++kx) {
                int xx = x + kx - radius;
                if (xx < 0 || xx >= width) continue;
                int pix = ((yy * width + xx) * 3) + c;
                acc += (float)in[pix] * cKernel[ky * ksize + kx];
            }
        }
        int value = (int)lrintf(acc);
        if (value < 0) value = 0;
        if (value > 255) value = 255;
        out[((y * width + x) * 3) + c] = (unsigned char)value;
    }
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 7) {
        fprintf(stderr, "Usage: %s image.ppm kernel.txt output.ppm partitions [block_x block_y]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *image_path = argv[1];
    const char *kernel_path = argv[2];
    const char *out_path = argv[3];
    int partitions = atoi(argv[4]);
    int block_x = (argc == 7) ? atoi(argv[5]) : 16;
    int block_y = (argc == 7) ? atoi(argv[6]) : 16;

    if (partitions <= 0) partitions = 1;
    if (block_x <= 0 || block_y <= 0 || block_x * block_y > 1024) {
        fprintf(stderr, "Invalid CUDA block size\n");
        return EXIT_FAILURE;
    }

    double t0 = seconds_now();
    Image img = read_ppm(image_path);
    Kernel ker = read_kernel(kernel_path);
    if (ker.size * ker.size > MAX_KERNEL_VALUES) {
        fprintf(stderr, "Kernel too large for constant memory buffer\n");
        return EXIT_FAILURE;
    }

    size_t bytes = (size_t)img.width * (size_t)img.height * 3u;
    Image out = {img.width, img.height, 255, (unsigned char*)calloc(bytes, 1)};
    if (!out.rgb) { fprintf(stderr, "Out of memory for output image\n"); return EXIT_FAILURE; }

    unsigned char *d_in = NULL;
    unsigned char *d_out = NULL;

    double t_transfer_h2d_start = seconds_now();
    CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, img.rgb, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));
    CHECK_CUDA(cudaMemcpyToSymbol(cKernel, ker.values, (size_t)ker.size * (size_t)ker.size * sizeof(float)));
    double t_transfer_h2d = seconds_now() - t_transfer_h2d_start;

    cudaEvent_t ev_start, ev_end;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_end));
    CHECK_CUDA(cudaEventRecord(ev_start));

    dim3 block((unsigned int)block_x, (unsigned int)block_y);
    int rows_per_part = (img.height + partitions - 1) / partitions;

    for (int p = 0; p < partitions; ++p) {
        int y0 = p * rows_per_part;
        int y1 = y0 + rows_per_part;
        if (y1 > img.height) y1 = img.height;
        if (y0 >= y1) continue;

        dim3 grid((img.width + block.x - 1) / block.x,
                  (y1 - y0 + block.y - 1) / block.y);

        convolution_kernel_rgb<<<grid, block>>>(d_in, d_out, img.width, img.height, ker.size, y0, y1);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaEventRecord(ev_end));
    CHECK_CUDA(cudaEventSynchronize(ev_end));
    float kernel_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, ev_start, ev_end));

    double t_transfer_d2h_start = seconds_now();
    CHECK_CUDA(cudaMemcpy(out.rgb, d_out, bytes, cudaMemcpyDeviceToHost));
    double t_transfer_d2h = seconds_now() - t_transfer_d2h_start;

    write_ppm(out_path, &out);
    double t1 = seconds_now();

    printf("Image: %s\n", image_path);
    printf("Kernel: %s\n", kernel_path);
    printf("Output: %s\n", out_path);
    printf("Image size: %d x %d\n", img.width, img.height);
    printf("Kernel size: %d x %d\n", ker.size, ker.size);
    printf("Partitions: %d\n", partitions);
    printf("CUDA block size: %d x %d\n", block_x, block_y);
    printf("CUDA H2D setup/transfer time: %.6f seconds\n", t_transfer_h2d);
    printf("CUDA kernel time: %.6f seconds\n", kernel_ms / 1000.0f);
    printf("CUDA D2H transfer time: %.6f seconds\n", t_transfer_d2h);
    printf("Total elapsed time: %.6f seconds\n", t1 - t0);

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_end));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(img.rgb);
    free(out.rgb);
    free(ker.values);

    return EXIT_SUCCESS;
}

NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_60
TARGET = cudaconv
SRC = src/convolution_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET) *.o *.ppm *.out *.err

MPICC ?= mpicc
CFLAGS ?= -O3 -fopenmp -Wall -Wextra
LDFLAGS ?= -lm
TARGET = hybridconv
SRC = src/convolution_hybrid.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o *.ppm *.out *.err

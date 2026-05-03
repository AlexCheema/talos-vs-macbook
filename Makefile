CC       ?= clang
NVCC     ?= nvcc
CFLAGS   ?= -O3 -march=native -ffast-math -Wall -Wextra -Wno-unused-parameter
NVFLAGS  ?= -O3 -arch=sm_121

UNAME_S  := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
HAVE_NVCC :=
LDLIBS   :=
else
HAVE_NVCC := $(shell command -v $(NVCC) 2>/dev/null)
# glibc auto-vectorizes expf() under -O3 -ffast-math via libmvec.
LDLIBS   := -lm -lmvec
endif

all: bench_c bench_c_q412 bench_c_batch
ifneq ($(HAVE_NVCC),)
all: bench_cuda bench_cuda_persistent
endif

bench_c: bench_c.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

bench_c_q412: bench_c_q412.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

bench_cuda: bench_cuda.cu
	$(NVCC) $(NVFLAGS) $< -o $@

bench_cuda_persistent: bench_cuda_persistent.cu
	$(NVCC) $(NVFLAGS) $< -o $@

bench_c_batch: bench_c_batch.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f bench_c bench_c_q412 bench_c_batch bench_cuda bench_cuda_persistent

.PHONY: all clean

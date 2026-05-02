CC      ?= clang
CFLAGS  ?= -O3 -march=native -ffast-math -Wall -Wextra -Wno-unused-parameter

all: bench_c bench_c_q412

bench_c: bench_c.c
	$(CC) $(CFLAGS) $< -o $@

bench_c_q412: bench_c_q412.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f bench_c bench_c_q412

.PHONY: all clean

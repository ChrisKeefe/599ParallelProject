CC = nvcc
CFLAGS = -Xcompiler -fopenmp
# TODO: replace GPUCAP with architecture-specific values. 60 covers P100
GPUCAP = 60
ARCHFLAGS = -arch=compute_${GPUCAP} -code=sm_${GPUCAP}
CFLAGS = -Wall
LFLAGS = -lm -lcuda

.phony = clean run time

elkan_cuda:
	module load cuda
	${CC} ${ARCHFLAGS} ${CFLAGS} ${LFLAGS} elkan_cuda.c csvparser.c -o elkan_cuda

# Use like this: `make run K=5`
run: elkan_cuda
	./elkan_cuda $k $K "../data/pairs.csv" "," 0 0

clean:
	rm elkan_cuda

time: elkan_cuda
	./elkan_cuda 5 "../data/pairs.csv" "," 0 0
	./elkan_cuda 10 "../data/pairs.csv" "," 0 0
	./elkan_cuda 15 "../data/pairs.csv" "," 0 0
	./elkan_cuda 25 "../data/pairs.csv" "," 0 0

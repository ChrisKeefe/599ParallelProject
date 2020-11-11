CC = nvcc
# TODO: replace compute_X and sm_X with architecture-specific vals
CFLAGS = -Wall -arch=compute_X -code=sm_X
LFLAGS = -lm -lcuda

.phony = clean run time

elkan_cuda:
	${CC} ${CFLAGS} elkan_cuda.c csvparser.c -o elkan_cuda ${LFLAGS}

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

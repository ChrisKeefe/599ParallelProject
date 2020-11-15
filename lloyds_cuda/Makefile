CC = nvcc
# TODO: replace compute_X and sm_X with architecture-specific vals
CFLAGS = -Wall -arch=compute_X -code=sm_X
LFLAGS = -lm -lcuda

.phony = clean run time

lloyds_cuda:
	${CC} ${CFLAGS} lloyds_cuda.c csvparser.c -o lloyds_cuda ${LFLAGS}

# Use like this: `make run K=5`
run: lloyds_cuda
	./lloyds_cuda $k $K "../data/pairs.csv" "," 0 0

clean:
	rm lloyds_cuda

time: lloyds_cuda
	./lloyds_cuda 5 "../data/pairs.csv" "," 0 0
	./lloyds_cuda 10 "../data/pairs.csv" "," 0 0
	./lloyds_cuda 15 "../data/pairs.csv" "," 0 0
	./lloyds_cuda 25 "../data/pairs.csv" "," 0 0

CC = nvcc
# TODO: replace GPUCAP with architecture-specific values. 60 covers P100
GPUCAP = 60
ARCHFLAGS = -arch=compute_${GPUCAP} -code=sm_${GPUCAP}
CFLAGS = -Wall
LFLAGS = -lm -lcuda

.phony = clean run time

lloyds_cuda:
	module load cuda
	${CC} ${ARCHFLAGS} ${CFLAGS} ${LFLAGS} lloyds_cuda.c csvparser.c -o lloyds_cuda

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

CC = nvcc
# TODO: replace GPUCAP with architecture-specific values. 60 covers P100
GPUCAP = 60
ARCHFLAGS = -arch=compute_${GPUCAP} -code=sm_${GPUCAP}
CFLAGS = -Wall
LFLAGS = -lm -lcuda

.phony = clean run test time

lloyds_cuda:
	module load cuda
	${CC} ${ARCHFLAGS} ${CFLAGS} ${LFLAGS} lloyds_cuda.c csvparser.c -o lloyds_cuda

run: lloyds_cuda
	./jobscripts/lloyds_cuda_run.sh

test: lloyds_cuda
	./jobscripts/lloyds_cuda_test.sh

time: lloyds_cuda
	./jobscripts/lloyds_cuda_time.sh

clean:
	rm lloyds_cuda

CC = nvcc
CFLAGS = -Xcompiler -fopenmp
# TODO: replace GPUCAP with architecture-specific values. 60 covers P100
GPUCAP = 60
ARCHFLAGS = -arch=compute_${GPUCAP} -code=sm_${GPUCAP}
LFLAGS = -lm -lcuda
OPTFLAGS = -Xcompiler -fopenmp

.phony = clean run test time

elkan_cuda:
	${CC} ${ARCHFLAGS} ${CFLAGS} ${LFLAGS} ${OPTFLAGS} elkan_cuda.cu csvparser.c -o elkan_cuda

run: elkan_cuda
	./jobscripts/elkan_cuda_run.sh

test: elkan_cuda
	./jobscripts/elkan_cuda_test.sh

time: elkan_cuda
	./jobscripts/elkan_cuda_time.sh

clean:
	rm elkan_cuda

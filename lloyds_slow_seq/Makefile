CC = gcc
CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp

.phony = clean run

lloyds_slow_seq:
	${CC} ${CFLAGS} lloyds_slow_seq.c csvparser.c -o lloyds_slow_seq ${LFLAGS} -fopenmp -O3

# Use like this: `make run K=5`
run: lloyds_slow_seq
	./lloyds_slow_seq $k $K "../data/iris.csv" "," 1 1

clean:
	rm lloyds_slow_seq

time: lloyds_slow_seq
	./lloyds_slow_seq 5 "../data/pairs.csv" "," 0 0
	./lloyds_slow_seq 10 "../data/pairs.csv" "," 0 0
	./lloyds_slow_seq 15 "../data/pairs.csv" "," 0 0
	./lloyds_slow_seq 25 "../data/pairs.csv" "," 0 0
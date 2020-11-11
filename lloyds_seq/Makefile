CC = gcc
CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp

.phony = clean run time

lloyds_seq:
	${CC} ${CFLAGS} lloyds_seq.c csvparser.c -o lloyds_seq ${LFLAGS} ${OPTFLAGS} -O3

# Use like this: `make run K=5`
run: lloyds_seq
	./lloyds_seq $k $K "../data/iris.csv" "," 1 1

test: lloyds_seq
	./lloyds_seq 3 "../data/iris.csv" "," 1 1 1

clean:
	rm lloyds_seq

time: lloyds_seq
	./lloyds_seq 5 "../data/pairs.csv" "," 0 0
	./lloyds_seq 10 "../data/pairs.csv" "," 0 0
	./lloyds_seq 15 "../data/pairs.csv" "," 0 0
	./lloyds_seq 25 "../data/pairs.csv" "," 0 0

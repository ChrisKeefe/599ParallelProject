CC = gcc
CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp -pthread

.phony = clean run

lloyds_seq:
	${CC} ${CFLAGS} lloyds_seq.c csvparser.c -o lloyds_seq ${LFLAGS}

# Use like this: `make run K=5`
run: lloyds_seq
	./lloyds_seq $k $K "../data/iris.csv" "," 1 1

clean:
	rm lloyds_seq

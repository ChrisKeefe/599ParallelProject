CC = gcc
CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp -pthread

.phony = clean run

lloyds_seq:
	${CC} ${CFLAGS} lloyds_seq.c -o lloyds_seq ${LFLAGS}

run: lloyds_seq
	./lloyds_seq

clean:
	rm lloyds_seq

CC = gcc
CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp -pthread

.phony = clean

lloyds_seq:
	${CC} ${CFLAGS} lloyds_seq.c -o lloyds_seq ${LFLAGS}

clean:
	rm lloyds_seq

CC = gcc
CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp

.phony = all

all:
	${CC} ${CFLAGS} lloyds_par.c csvparser.c -o lloyds_par1 ${LFLAGS} ${OPTFLAGS} -O3
	${CC} ${CFLAGS} lloyds_par8.c csvparser.c -o lloyds_par8 ${LFLAGS} ${OPTFLAGS} -O3
	${CC} ${CFLAGS} lloyds_par16.c csvparser.c -o lloyds_par16 ${LFLAGS} ${OPTFLAGS} -O3
	${CC} ${CFLAGS} lloyds_par24.c csvparser.c -o lloyds_par24 ${LFLAGS} ${OPTFLAGS} -O3
	${CC} ${CFLAGS} lloyds_par32.c csvparser.c -o lloyds_par32 ${LFLAGS} ${OPTFLAGS} -O3

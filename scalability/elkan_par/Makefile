CC = gcc
CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp

.phony = all

all:
	${CC} ${CFLAGS} elkan_par.c csvparser.c -o elkan_par1 ${LFLAGS} ${OPTFLAGS} -O3
	${CC} ${CFLAGS} elkan_par8.c csvparser.c -o elkan_par8 ${LFLAGS} ${OPTFLAGS} -O3
	${CC} ${CFLAGS} elkan_par16.c csvparser.c -o elkan_par16 ${LFLAGS} ${OPTFLAGS} -O3
	${CC} ${CFLAGS} elkan_par24.c csvparser.c -o elkan_par24 ${LFLAGS} ${OPTFLAGS} -O3
	${CC} ${CFLAGS} elkan_par32.c csvparser.c -o elkan_par32 ${LFLAGS} ${OPTFLAGS} -O3

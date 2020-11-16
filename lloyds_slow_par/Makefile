CC = gcc
CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp

.phony = clean run

lloyds_slow_par:
	${CC} ${CFLAGS} lloyds_slow_par.c csvparser.c -o lloyds_slow_par ${LFLAGS} -fopenmp -O3

# Use like this: `make run K=5`
run: lloyds_slow_par
	./lloyds_slow_par $k $K "../data/pairs.csv" "," 0 0

clean:
	rm lloyds_slow_par

time: lloyds_slow_par
	./lloyds_slow_par 5 "../data/pairs.csv" "," 0 0
	./lloyds_slow_par 10 "../data/pairs.csv" "," 0 0
	./lloyds_slow_par 15 "../data/pairs.csv" "," 0 0
	./lloyds_slow_par 25 "../data/pairs.csv" "," 0 0
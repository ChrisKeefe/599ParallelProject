CC = gcc
CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp

.phony = clean run time

lloyds_par:
	${CC} ${CFLAGS} lloyds_par.c csvparser.c -o lloyds_par ${LFLAGS} ${OPTFLAGS} -O3

# Use like this: `make run K=5`
run: lloyds_par
	./lloyds_par $k $K "../data/pairs.csv" "," 0 0

# Adding another argument to the end of the call runs in debug mode with our preselected iris centers
test: lloyds_par
	./lloyds_par 3 "../data/iris.csv" "," 1 1 1

clean:
	rm lloyds_par

time: lloyds_par
	./lloyds_par 5 "../data/pairs.csv" "," 0 0
	./lloyds_par 10 "../data/pairs.csv" "," 0 0
	./lloyds_par 15 "../data/pairs.csv" "," 0 0
	./lloyds_par 25 "../data/pairs.csv" "," 0 0

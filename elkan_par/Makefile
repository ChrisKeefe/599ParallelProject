CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp

.phony = clean run time

elkan_par:
	${CC} ${CFLAGS} elkan_par.c csvparser.c -o elkan_par ${LFLAGS} ${OPTFLAGS} -O3

# Use like this: `make run K=5`
run: elkan_par
	./elkan_par $k $K "../data/pairs.csv" "," 0 0

test: elkan_par
	./elkan_par 3 "../data/iris.csv" "," 1 1 1

clean:
	rm elkan_par

time: elkan_par
	./elkan_par 5 "../data/pairs.csv" "," 0 0
	./elkan_par 10 "../data/pairs.csv" "," 0 0
	./elkan_par 15 "../data/pairs.csv" "," 0 0
	./elkan_par 25 "../data/pairs.csv" "," 0 0

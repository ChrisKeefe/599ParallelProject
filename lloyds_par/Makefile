CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp

.phony = clean run time

lloyds_par:
	${CC} ${CFLAGS} lloyds_par.c csvparser.c -o lloyds_par ${LFLAGS} ${OPTFLAGS}

# Use like this: `make run K=5`
run: lloyds_par
	./lloyds_par $k $K "../data/pairs.csv" "," 0 0

clean:
	rm lloyds_par

time: lloyds_par
	./lloyds_par 5 "../data/pairs.csv" "," 0 0
	./lloyds_par 10 "../data/pairs.csv" "," 0 0
	./lloyds_par 15 "../data/pairs.csv" "," 0 0
	./lloyds_par 25 "../data/pairs.csv" "," 0 0

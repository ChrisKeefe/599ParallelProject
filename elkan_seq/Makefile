CFLAGS = -Wall
LFLAGS = -lm
OPTFLAGS = -fopenmp

.phony = clean run time

elkan_seq:
	${CC} ${CFLAGS} elkan_seq.c csvparser.c -o elkan_seq ${LFLAGS} ${OPTFLAGS}

# Use like this: `make run K=5`
run: elkan_seq
	./elkan_seq $k $K "../data/pairs.csv" "," 0 0

clean:
	rm elkan_seq

time: elkan_seq
	./elkan_seq 5 "../data/pairs.csv" "," 0 0
	./elkan_seq 10 "../data/pairs.csv" "," 0 0
	./elkan_seq 15 "../data/pairs.csv" "," 0 0
	./elkan_seq 25 "../data/pairs.csv" "," 0 0

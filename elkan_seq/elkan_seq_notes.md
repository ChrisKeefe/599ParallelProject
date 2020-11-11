Find the distance between centers. If d(c1, c2) >= 2d(pt, c1), pt is closer to c1, and we don't calculate d(pt, c2)

----------------------------------------------------------------------

upper bound: smallest distance between a poind and all centers that have been checked so far
lower bound:

~~1. Pick initial centers randomly from our data (double[K][dim(X)])~~
~~2. Set lower bound to 0 for each (pt, ctr) pair (double[N][K])~~
~~3. Calculate inter-center distances. (double arr[K])~~
4. Check each point against all centers where d(c1, c2) < 2(upper_bound)
5. If a distance calculation is required, perform it and assign the result to the lower bound for that pt/ctr pair

TODO: Add compiler optimization flags to makefiles
TODO: Output clusterings?
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// Program should take K and then a data set
int main(int argc, char *argv[]){
printf("%d", argc);

// TODO: bump this up to 3 when we're ready to deal with data
if(argc != 2){
    printf("Incorrect number of args. Should be 2, received %d\n", argc - 1);
    exit(1);
}

int K = atoi(argv[1]);
int clustering[K];

exit 0;
}
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "csvparser.h"

// Program should take K, a data set (.csv), a delimiter,
// and a binary flag data_contains_header
int main(int argc, char *argv[]){
  CsvParser *reader;
  CsvRow *row;
  const CsvRow *header;

  printf("%d\n", argc);
  if(argc != 5){
      printf("Incorrect number of args. Should be 4, received %d\n", argc - 1);
      exit(1);
  }

  int K = atoi(argv[1]);
  int clustering[K];
  char *delimiter = argv[3];
  int has_header_row = atoi(argv[4]);

  // Take in data set
  reader = CsvParser_new(argv[2], delimiter, has_header_row);
  header = CsvParser_getHeader(reader);

  const char **headerFields = CsvParser_getFields(header);
  printf("%s\n", headerFields[0]);


  exit(0);
}
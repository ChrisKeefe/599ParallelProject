#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "csvparser.h"

// Program should take K, a data set (.csv), a delimiter,
// a binary flag data_contains_header, and a binary flag to drop labels
int main(int argc, char *argv[]){
  CsvParser *reader;
  CsvRow *row;

  if(argc != 6){
      printf("Incorrect number of args. Should be 5, received %d\n", argc - 1);
      exit(1);
  }

  int K = atoi(argv[1]);
  int clustering[K];
  char *data_fp = argv[2];
  char *delimiter = argv[3];
  int has_header_row = atoi(argv[4]);
  int drop_labels = atoi(argv[5]);

  // Take in data set
  reader = CsvParser_new(data_fp, delimiter, has_header_row);

  // Get number of columns
  row = CsvParser_getRow(reader);
  int num_cols = CsvParser_getNumFields(row);
  CsvParser_destroy_row(row);

  if (drop_labels){
    num_cols--;
  }

  // Get number of rows like lazy people
  int num_rows = 1;
  while ((row = CsvParser_getRow(reader))){
    num_rows++;
    CsvParser_destroy_row(row);
  }

  // printf("Num cols: %d\tNum rows: %d\n", num_cols, num_rows);

  float data_matrix[num_rows][num_cols];

  // Torch the CsvParser and start again so we can read data in.
  CsvParser_destroy(reader);

  reader = CsvParser_new(data_fp, delimiter, has_header_row);

  int row_index = 0;
  int col_index;
  while ((row = CsvParser_getRow(reader))){
    const char **row_fields = CsvParser_getFields(row);

    for (col_index = 0; col_index < num_cols; col_index++) {
      data_matrix[row_index][col_index] = atof(row_fields[col_index]);
    }

    CsvParser_destroy_row(row);
    row_index++;
  }

  CsvParser_destroy(reader);

  // Initialize some cluster centers from random rows in our data

  // Assign points to cluster centers
    // calculate within-cluster sum of squares
    // total within-cluster sum of squares

  // break out of loop if total within-cluster sum of squares has converged

  // Find cluster means and reassign centers

  exit(0);
}
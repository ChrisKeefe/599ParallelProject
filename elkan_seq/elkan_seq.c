#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>

#include "csvparser.h"

void vector_init(double *a, int length) {
  for (int i = 0; i < length; i++) {
    a[i] = 0;
  }
}

void vector_copy(double *dst, double *src, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = src[i];
  }
}

void vector_add(double *dst, double *a, double *b, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = a[i] + b[i];
  }
}

void vector_elementwise_avg(double *dst, double *a, int denominator, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = a[i] / denominator;
  }
}

double vector_L2_norm(double *a, int length) {
  double vec_norm = 0;

  for (int i = 0; i < length; i++) {
    vec_norm += a[i] * a[i];
  }

  return vec_norm;
}

void vector_sub(double *dst, double *a, double *b, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = a[i] - b[i];
  }
}

// Program should take K, a data set (.csv), a delimiter,
// a binary flag data_contains_header, and a binary flag to drop labels
int main(int argc, char *argv[]){
  // Seed for consistent cluster center selection
  // In a working implementation, seeding would be variable (e.g. time(NULL))
  srand(111);
  CsvParser *reader;
  CsvRow *row;
  int i, j;

  if(argc != 6){
      printf("Incorrect number of args. Should be 5, received %d\n", argc - 1);
      exit(1);
  }

  int K = atoi(argv[1]);
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

  // Torch the CsvParser and start again so we can read data in.
  CsvParser_destroy(reader);

  reader = CsvParser_new(data_fp, delimiter, has_header_row);

  double **data_matrix = malloc(num_rows * sizeof(double *));
  for (i = 0; i < num_rows; i++) {
    data_matrix[i] = malloc(num_cols * sizeof(double));
  }


  int row_index = 0;
  while ((row = CsvParser_getRow(reader))){
    const char **row_fields = CsvParser_getFields(row);

    for (int col_index = 0; col_index < num_cols; col_index++) {
      data_matrix[row_index][col_index] = atof(row_fields[col_index]);
    }

    CsvParser_destroy_row(row);
    row_index++;
  }

  CsvParser_destroy(reader);

  // Initialize some cluster centers from random rows in our data
  // Given the fact that we will usually have way more rows than centers, we can
  // probably just roll a number and reroll if we already rolled it. Collisions
  // should be relatively infrequent
  bool collided;
  double centers[K][num_cols];
  for (i = 0; i < K; i++) {
    int center_indices[K];
    collided = true;

    while (collided) {
      center_indices[i] = rand() % num_rows;
      collided = false;

      for (j = 0; j < i; j++) {
        if (center_indices[j] == center_indices[i]) {
          collided = true;
          break;
        }
      }

      vector_copy(centers[i], data_matrix[center_indices[i]], num_cols);
    }
  }

  // These are for testing against R with iris data
  // int center_indices[3] = {12, 67, 106};
  // for (i = 0; i < K; i ++) {
  //   vector_copy(centers[i], data_matrix[center_indices[i]], num_cols);
  // }

  printf("Initial cluster centers:\n");
  for (i = 0; i < K; i++) {
    for (j = 0; j < num_cols; j++) {
      printf("%f ", centers[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  int num_iterations = 0;
  int *clusterings = calloc(num_rows, sizeof(int));
  double *l_bounds = calloc(num_rows * K, sizeof(double));
  double *u_bounds = calloc(num_rows, sizeof(double));
  double *ctr_ctr_dists = malloc(K * K * sizeof(double));
  // These need better names
  double s[K];
  bool *r = calloc(num_rows, sizeof(bool));
  bool changes;

  double tstart = omp_get_wtime();
  // TODO: implement algo here

  // Check each point against all centers where d(c1, c2) < 2 * upper_bound)
  int this_ctr, this_pt;
  // we assume each point is assigned to center 0
  // we check distance from center 1, 2, etc to current center against upper bound
  // (Upper bound is the distance to the best-center-so-far)
  double tmp_diff[num_cols];
  double min_diff = INFINITY;
  for (this_pt = 0; this_pt < num_rows; this_pt++) {
    // find the distance from this point to center_0
    vector_sub(tmp_diff, centers[0], data_matrix[this_pt], num_cols);
    u_bounds[this_pt] = vector_L2_norm(tmp_diff, num_cols);
    l_bounds[this_pt * K + 0];

    for (this_ctr = 1; this_ctr < K; this_ctr++) {
      if (ctr_ctr_dists[clusterings[this_pt] + this_ctr * K] < 2 * u_bounds[this_pt]){
        vector_sub(tmp_diff, centers[this_ctr], data_matrix[this_pt], num_cols);

        // Capture the lower bound, assign a new clustering & u_bound if closer
        l_bounds[this_pt * K + this_ctr] = vector_L2_norm(tmp_diff, num_cols);

        if (l_bounds[this_pt * K + this_ctr] < u_bounds[this_pt]) {
          clusterings[this_pt] = this_ctr;
          u_bounds[this_pt] = l_bounds[this_pt * K + this_ctr];
        }
      }
    }
  }

  bool r = false;
  while(1) {
    // Calculate initial center-center distances
    // TODO: reduce number of distance calculations
    for (i = 0; i < K; i++) {
      for (j = 0; j < K; j++) {
        vector_sub(tmp_diff, centers[i], centers[j], num_cols);
        ctr_ctr_dists[i * K + j] = vector_L2_norm(tmp_diff, num_cols);

        if (ctr_ctr_dists[i * K + j] < min_diff) {
          min_diff = ctr_ctr_dists[i * K + j];
        }
      }

      s[i] = min_diff / 2;
    }
  
    for (this_pt = 0; this_pt < num_rows; this_pt++) {
      if (u_bounds[this_pt] > s[clusterings[this_pt]]) {
        for(this_ctr = 0; this_ctr < K; this_ctr++) {

        }
      }
    }
  }

  double tend = omp_get_wtime();

  printf("\nCenter-center distances:\n");
  for (i = 0; i < K; i++) {
    for (j = 0; j < num_cols; j++) {
      printf("%f ", ctr_ctr_dists[j + i * K]);
    }
    printf("\n");
  }

  printf("\nFinal cluster centers:\n");
  for (i = 0; i < K; i++) {
    for (j = 0; j < num_cols; j++) {
      printf("%f ", centers[i][j]);
    }
    printf("\n");
  }

  printf("\nNum iterations: %d\n", num_iterations);
  printf("Time taken for %d clusters: %f seconds\n", K, tend - tstart);

  for (i = 0; i < num_rows; i++) {
    free(data_matrix[i]);
  }

  free(data_matrix);
  free(clusterings);
  exit(0);
}

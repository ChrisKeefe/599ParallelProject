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

void vector_subtract(double *dst, double *a, double *b, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = a[i] - b[i];
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
  double vec_length = 0;

  for (int i = 0; i < length; i++) {
    vec_length += a[i] * a[i];
  }

  return sqrt(vec_length);
}

double vector_sum(double *a, int length) {
  double sum = 0;

  for (int i = 0; i < length; i ++) {
    sum += a[i];
  }

  return sum;
}

void vector_square(double *dst, double *a, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = a[i] * a[i];
  }
}

// Program should take K, a data set (.csv), a delimiter,
// a binary flag data_contains_header, and a binary flag to drop labels
int main(int argc, char *argv[]){
  srand(111);
  CsvParser *reader;
  CsvRow *row;

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
  for (int i = 0; i < num_rows; i++) {
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
  for (int i = 0; i < K; i++) {
    int center_indices[K];
    collided = true;

    while (collided) {
      center_indices[i] = rand() % num_rows;
      collided = false;

      for (int j = 0; j < i; j++) {
        if (center_indices[j] == center_indices[i]) {
          collided = true;
          break;
        }
      }

      vector_copy(centers[i], data_matrix[center_indices[i]], num_cols);
    }
  }

  printf("Initial cluster centers:\n");
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < num_cols; j++) {
      printf("%f ", centers[i][j]);
    }
    printf("\n");
  }

  int num_iterations = 0;

  int *clusterings = calloc(num_rows, sizeof(int));
  double *cluster_avg = malloc(num_rows * sizeof(double));

  bool changes;
  double tstart = omp_get_wtime();
  while (1) {
    changes = false;

    omp_set_num_threads(32);
    // Assign points to cluster centers
    int center, observation, arg_min;
    double min_norm, local_norm, diff[num_cols];
    #pragma omp parallel for \
      private(center, observation, min_norm, local_norm, diff, arg_min) \
      shared(num_rows, K, data_matrix, centers)
    for (observation = 0; observation < num_rows; observation++) {
      arg_min = 0;
      min_norm = -1;

      for (center = 0; center < K; center++) {

        vector_subtract(diff, data_matrix[observation], centers[center], num_cols);
        local_norm = vector_L2_norm(diff, num_cols);
        if ((min_norm == -1) || (local_norm < min_norm)) {
          arg_min = center;
          min_norm = local_norm;
        }
      }

      if (clusterings[observation] != arg_min) {
        changes = true;
      }
      clusterings[observation] = arg_min;
    }

    // break out of loop if total within-cluster sum of squares has converged
    if (!changes) {
      break;
    }
    num_iterations++;

    // Find cluster means and reassign centers
    int cluster_index, element, elements_in_cluster;
    #pragma omp parallel for \
      private(cluster_index, element, elements_in_cluster) \
      shared(num_rows, clusterings, data_matrix, K)
    for (cluster_index = 0; cluster_index < K; cluster_index++) {
      elements_in_cluster = 0;
      vector_init(cluster_avg, num_rows);

      for (element = 0; element < num_rows; element++) {
        if (clusterings[element] == cluster_index) {
          vector_add(cluster_avg, cluster_avg, data_matrix[element], num_cols);
          elements_in_cluster++;
        }
      }

      vector_elementwise_avg(cluster_avg, cluster_avg, elements_in_cluster, num_cols);
      vector_copy(centers[cluster_index], cluster_avg, num_cols);
    }
  }
  double tend = omp_get_wtime() - tstart;

  printf("\nFinal cluster centers:\n");
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < num_cols; j++) {
      printf("%f ", centers[i][j]);
    }
    printf("\n");
  }

  printf("\nNum iterations: %d\n", num_iterations);
  printf("Time taken %f seconds\n", tend);

  for (int i = 0; i < num_rows; i++) {
    free(data_matrix[i]);
  }
  free(data_matrix);

  free(clusterings);
  free(cluster_avg);

  exit(0);
}

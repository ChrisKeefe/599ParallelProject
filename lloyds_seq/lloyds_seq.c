#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "csvparser.h"

void vector_init(float *a, int length) {
  for (int i = 0; i < length; i++) {
    a[i] = 0;
  }
}

void vector_copy(float *dst, float *src, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = src[i];
  }
}

void vector_subtract(float *dst, float *a, float *b, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = a[i] - b[i];
  }
}

void vector_add(float *dst, float *a, float *b, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = a[i] + b[i];
  }
}

void vector_elementwise_avg(float *dst, float *a, int denominator, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = a[i] / denominator;
  }
}

float vector_length(float *a, int length) {
  float vec_length = 0;

  for (int i = 0; i < length; i++) {
    vec_length += a[i] * a[i];
  }

  return sqrt(vec_length);
}

float vector_sum(float *a, int length) {
  float sum = 0;

  for (int i = 0; i < length; i ++) {
    sum += a[i];
  }

  return sum;
}

void vector_square(float *dst, float *a, int length) {
  for (int i = 0; i < length; i++) {
    dst[i] = a[i] * a[i];
  }
}

// Program should take K, a data set (.csv), a delimiter,
// a binary flag data_contains_header, and a binary flag to drop labels
int main(int argc, char *argv[]){
  srand(time(NULL));
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

  float **data_matrix = malloc(num_rows * sizeof(float *));
  for (int i = 0; i < num_rows; i++) {
    data_matrix[i] = malloc(num_cols * sizeof(float));
  }


  int row_index = 0;
  while ((row = CsvParser_getRow(reader))){
    // printf("%d\n", row_index);
    const char **row_fields = CsvParser_getFields(row);
    // printf("%s\n", row_fields[0]);

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
  printf("Before init centers\n");
  bool collided;
  float centers[K][num_cols];
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
  printf("After init centers\n");

  printf("Initial cluster centers:\n");
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < num_cols; j++) {
      printf("%f ", centers[i][j]);
    }
    printf("\n");
  }

  float tot_withinss = -1;
  float last_tot_withinss;
  int num_iterations = 0;
  while (1) {
    last_tot_withinss = tot_withinss;

    float withinss[K];
    vector_init(withinss, K);

    float size[K];
    int cluster[num_rows];

    // Assign points to cluster centers
    for (int observation = 0; observation < num_rows; observation++) {
      float min_norm = -1;
      int arg_min;

      for (int center = 0; center < K; center++) {
        float diff[num_cols];

        vector_subtract(diff, data_matrix[observation], centers[center], num_cols);
        float local_norm = vector_length(diff, num_cols);
        if ((min_norm == -1) || (local_norm < min_norm)) {
          arg_min = center;
          min_norm = local_norm;
        }
      }

      cluster[observation] = arg_min;

      // calculate within-cluster sum of squares
      float ss[num_rows];
      vector_subtract(ss, data_matrix[observation], centers[arg_min], num_cols);
      vector_square(ss, ss, num_cols);
      float obs_local_ss = vector_sum(ss, num_cols);

      withinss[arg_min] += obs_local_ss;
      size[arg_min]++;
    }

    // total within-cluster sum of squares
    tot_withinss = vector_sum(withinss, num_cols);
    num_iterations++;

    // break out of loop if total within-cluster sum of squares has converged
    if (tot_withinss == last_tot_withinss) {
      break;
    }

    // Find cluster means and reassign centers
    for (int cluster_index = 0; cluster_index < K; cluster_index++) {
      int elements_in_cluster = 0;
      float cluster_avg[num_rows];
      vector_init(cluster_avg, num_rows);

      for (int element = 0; element < num_rows; element++) {
        if (cluster[element] == cluster_index) {
          vector_add(cluster_avg, cluster_avg, data_matrix[element], num_cols);
          elements_in_cluster++;
        }
      }

      vector_elementwise_avg(cluster_avg, cluster_avg, elements_in_cluster, num_cols);
      vector_copy(centers[cluster_index], cluster_avg, num_cols);
    }
  }

  printf("\nNum iterations: %d\n\n", num_iterations);
  printf("Final cluster centers:\n");
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < num_cols; j++) {
      printf("%f ", centers[i][j]);
    }
    printf("\n");
  }

  for (int i = 0; i < num_rows; i++) {
    free(data_matrix[i]);
  }
  free(data_matrix);

  exit(0);
}

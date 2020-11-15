#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>

#include "csvparser.h"

void warmUpGPU();

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

// Program should take K, a data set (.csv), a delimiter,
// a binary flag data_contains_header, and a binary flag to drop labels
int main(int argc, char *argv[]) {
  // Seed for consistent cluster center selection
  // In a working implementation, seeding would be variable (e.g. time(NULL))
  srand(111);
  CsvParser *reader;
  CsvRow *row;
  int i, j;

  if(argc < 6) {
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
  double centers[K][num_cols];
  bool collided;

  if (argc == 7) {
    int center_indices[3] = {12, 67, 106};
    for (i = 0; i < K; i ++) {
      vector_copy(centers[i], data_matrix[center_indices[i]], num_cols);
    }
  } else {
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
  }

  printf("Initial cluster centers:\n");
  for (i = 0; i < K; i++) {
    for (j = 0; j < num_cols; j++) {
      printf("%f ", centers[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  int num_iterations = 0;
  int *clusterings = malloc(num_rows * sizeof(int));
  double cluster_means[num_cols];
  bool changes;

  double tstart = omp_get_wtime();

  // TODO: Look into how to memcpy a double pointer like this
  errCode=cudaMalloc((double***)&data_matrix, sizeof(unsigned int)*N);
  if(errCode != cudaSuccess) {
  cout << "\nError: A error with code " << errCode << endl;
  }

  errCode=cudaMalloc((unsigned int**)&clusterings, sizeof(nt)*num_rows;
  if(errCode != cudaSuccess) {
  cout << "\nError: B error with code " << errCode << endl;
  }

  errCode=cudaMalloc((unsigned int**)&dev_C, sizeof(unsigned int)*N);
  if(errCode != cudaSuccess) {
  cout << "\nError: C error with code " << errCode << endl;
  }

  //copy A to device
  errCode=cudaMemcpy( dev_A, A, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
  if(errCode != cudaSuccess) {
  cout << "\nError: A memcpy error with code " << errCode << endl;
  }

  //copy B to device
  errCode=cudaMemcpy( dev_B, B, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
  if(errCode != cudaSuccess) {
  cout << "\nError: A memcpy error with code " << errCode << endl;
  }

  //copy C to device (initialized to 0)
  errCode=cudaMemcpy( dev_C, C, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
  if(errCode != cudaSuccess) {
  cout << "\nError: A memcpy error with code " << errCode << endl;
  }

  while (1) {
    // Assign points to cluster centers
    changes = false;

    // If we didn't change any cluster assignments, we've reached convergence
    if (!changes) {
      break;
    }

    num_iterations++;

    // Find cluster means and reassign centers
    for (int cluster_index = 0; cluster_index < K; cluster_index++) {
      int elements_in_cluster = 0;
      vector_init(cluster_means, num_cols);

      for (int element = 0; element < num_rows; element++) {
        if (clusterings[element] == cluster_index) {
          vector_add(cluster_means, cluster_means, data_matrix[element], num_cols);
          elements_in_cluster++;
        }
      }

      vector_elementwise_avg(cluster_means, cluster_means, elements_in_cluster, num_cols);
      vector_copy(centers[cluster_index], cluster_means, num_cols);
    }
  }

  double tend = omp_get_wtime();

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

// We'll need to copy in data_matrix, centers, cluster_means, clusterings

// Borrowing from his G5_Q3, maybe we start with 1024 as block size. Calculate
// numblocks as ceil(N*1.0/1024)
__global__ void lloyds(double *cluster_means, double *centers, int *clusterings) {
  // we need to in some way parallelize over observations.
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Where N is num_rows
  if (tid >= N) {
    return;
  }

  // The above is taken directly from his G5_Q3 code

  int new_center;
  double best_diff = INFINITY;

  for (int center = 0; center < K; center++) {
    double current_diff = 0;
    double tmp;

    for (int col = 0; col < num_cols; col++) {
      tmp = data_matrix[tid][col] - centers[center][col];
      current_diff += tmp * tmp;
    }

    if (current_diff < best_diff) {
      best_diff = current_diff;
      new_center = center;
    }
  }

  if (clusterings[tid] != new_center) {
    // Make global?
    changes = true;
    clusterings[tid] = new_center;
  }
}


void warmUpGPU(){
  cudaDeviceSynchronize();
  return;
}

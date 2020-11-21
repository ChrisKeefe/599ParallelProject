#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>

#include "csvparser.h"

#define BLOCKSIZE 1024

using namespace std;

__global__ void lloyds(double *dev_data_matrix, double *dev_centers, int *dev_clusterings,
                       bool *dev_changes, int *dev_num_rows, int *dev_num_cols, int *dev_K);

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

  if (argc < 6) {
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

  double *data_matrix = (double *)malloc(num_rows * num_cols * sizeof(double));

  int row_index = 0;
  while ((row = CsvParser_getRow(reader))){
    const char **row_fields = CsvParser_getFields(row);

    for (int col_index = 0; col_index < num_cols; col_index++) {
      data_matrix[row_index * num_cols + col_index] = atof(row_fields[col_index]);
    }

    CsvParser_destroy_row(row);
    row_index++;
  }

  CsvParser_destroy(reader);

  // Initialize some cluster centers from random rows in our data
  // Given the fact that we will usually have way more rows than centers, we can
  // probably just roll a number and reroll if we already rolled it. Collisions
  // should be relatively infrequent
  double *centers = (double *)malloc(K * num_cols * sizeof(double));
  bool collided;

  if (argc == 7) {
    int center_indices[3] = {12, 67, 106};
    for (i = 0; i < K; i ++) {
      vector_copy(centers + i * num_cols, data_matrix + center_indices[i] * num_cols, num_cols);
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

        vector_copy(centers + i *num_cols, data_matrix + center_indices[i] * num_cols, num_cols);
      }
    }
  }

  printf("Initial cluster centers:\n");
  for (i = 0; i < K; i++) {
    for (j = 0; j < num_cols; j++) {
      printf("%f ", centers[i * num_cols + j]);
    }
    printf("\n");
  }
  printf("\n");

  int num_iterations = 0;
  int *clusterings = (int *)malloc(num_rows * sizeof(int));
  double cluster_means[num_cols];
  bool changes;

  double *dev_data_matrix;
  double *dev_centers;
  int *dev_clusterings;
  bool *dev_changes;
  int *dev_num_rows;
  int *dev_num_cols;
  int *dev_K;

  const unsigned int totalBlocks = ceil(num_rows * 1.0 / BLOCKSIZE);

  warmUpGPU();

  double kerneltime = 0;
  double transfertime = 0;
  double tstart = omp_get_wtime();

  cudaError_t errCode = cudaSuccess;
  double ttransferstart = omp_get_wtime();
  errCode = cudaMalloc(&dev_data_matrix, sizeof(double) * num_rows * num_cols);
  if (errCode != cudaSuccess) {
    cout << "\nError: data_matrix alloc error with code " << errCode << endl;
  }

  errCode = cudaMalloc(&dev_centers, sizeof(double) * K * num_cols);
  if (errCode != cudaSuccess) {
    cout << "\nError: centers alloc error with code " << errCode << endl;
  }

  errCode = cudaMalloc(&dev_clusterings, sizeof(int) * num_rows);
  if (errCode != cudaSuccess) {
    cout << "\nError: clusterings alloc error with code " << errCode << endl;
  }

  errCode = cudaMalloc(&dev_changes, sizeof(bool));
  if (errCode != cudaSuccess) {
    cout << "\nError: changes alloc error with code " << errCode << endl;
  }

  errCode = cudaMalloc(&dev_num_rows, sizeof(int));
  if (errCode != cudaSuccess) {
    cout << "\nError: num_rows alloc error with code " << errCode << endl;
  }

  errCode = cudaMemcpy(dev_num_rows, &num_rows, sizeof(int), cudaMemcpyHostToDevice);
  if (errCode != cudaSuccess) {
    cout << "\nError: num_rows memcpy error with code " << errCode << endl;
  }

  errCode = cudaMalloc(&dev_num_cols, sizeof(int));
  if (errCode != cudaSuccess) {
    cout << "\nError: num_cols alloc error with code " << errCode << endl;
  }

  errCode = cudaMemcpy(dev_num_cols, &num_cols, sizeof(int), cudaMemcpyHostToDevice);
  if (errCode != cudaSuccess) {
    cout << "\nError: num_cols memcpy error with code " << errCode << endl;
  }

  errCode = cudaMalloc(&dev_K, sizeof(int));
  if (errCode != cudaSuccess) {
    cout << "\nError: K alloc error with code " << errCode << endl;
  }

  errCode = cudaMemcpy(dev_K, &K, sizeof(int), cudaMemcpyHostToDevice);
  if (errCode != cudaSuccess) {
    cout << "\nError: K memcpy error with code " << errCode << endl;
  }
  transfertime += omp_get_wtime() - ttransferstart;

  while (1) {
    // Assign points to cluster centers
    changes = false;

    ttransferstart = omp_get_wtime();
    errCode = cudaMemcpy(dev_data_matrix, data_matrix, sizeof(double) * num_rows * num_cols, cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess) {
      cout << "\nError: data_matrix memcpy error with code " << errCode << endl;
    }

    errCode = cudaMemcpy(dev_centers, centers, sizeof(double) * K * num_cols, cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess) {
      cout << "\nError: centers memcpy error with code " << errCode << endl;
    }

    errCode = cudaMemcpy(dev_clusterings, clusterings, sizeof(int) * num_rows, cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess) {
      cout << "\nError: clusterings memcpy error with code " << errCode << endl;
    }

    errCode = cudaMemcpy(dev_changes, &changes, sizeof(bool), cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess) {
      cout << "\nError: changes memcpy error with code " << errCode << endl;
    }
    transfertime += omp_get_wtime() - ttransferstart;

    double kernelstart = omp_get_wtime();
    lloyds<<<totalBlocks, BLOCKSIZE>>>(dev_data_matrix, dev_centers, dev_clusterings, dev_changes, dev_num_rows, dev_num_cols, dev_K);
    cudaDeviceSynchronize();
    kerneltime += omp_get_wtime() - kernelstart;

    //copy data from device to host
    errCode = cudaMemcpy(centers, dev_centers, sizeof(double) * K * num_cols, cudaMemcpyDeviceToHost);
    if (errCode != cudaSuccess) {
      cout << "\nError: getting centers result from GPU error with code " << errCode << endl;
    }

    errCode = cudaMemcpy(clusterings, dev_clusterings, sizeof(int) * num_rows, cudaMemcpyDeviceToHost);
    if (errCode != cudaSuccess) {
      cout << "\nError: getting clusterings result from GPU error with code " << errCode << endl;
    }

    errCode = cudaMemcpy(&changes, dev_changes, sizeof(bool), cudaMemcpyDeviceToHost);
    if (errCode != cudaSuccess) {
      cout << "\nError: getting changes result from GPU error with code " << errCode << endl;
    }

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
          vector_add(cluster_means, cluster_means, data_matrix + element * num_cols, num_cols);
          elements_in_cluster++;
        }
      }

      vector_elementwise_avg(cluster_means, cluster_means, elements_in_cluster, num_cols);
      vector_copy(centers + cluster_index * num_cols, cluster_means, num_cols);
    }
  }

  double tend = omp_get_wtime();

  printf("\nFinal cluster centers:\n");
  for (i = 0; i < K; i++) {
    for (j = 0; j < num_cols; j++) {
      printf("%f ", centers[i * num_cols + j]);
    }
    printf("\n");
  }

  printf("\nNum iterations: %d\n", num_iterations);
  printf("Time taken for %d clusters: %f seconds\nkernel: %f seconds\n\n", K, tend - tstart, kerneltime);

  free(data_matrix);
  free(clusterings);

  exit(0);
}


// We'll need to copy in data_matrix, centers, clusterings

// Borrowing from his G5_Q3, maybe we start with 1024 as block size. Calculate
// numblocks as ceil(N*1.0/1024)
__global__ void lloyds(double *dev_data_matrix, double *dev_centers, int *dev_clusterings,
                       bool *dev_changes, int *dev_num_rows, int *dev_num_cols, int *dev_K) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= *dev_num_rows) {
    return;
  }

  int new_center;
  double best_diff = INFINITY;

  for (int center = 0; center < *dev_K; center++) {
    double current_diff = 0;
    double tmp;

    for (int col = 0; col < *dev_num_cols; col++) {
      tmp = dev_data_matrix[tid * *(dev_num_cols) + col] - dev_centers[center * *(dev_num_cols) + col];
      current_diff += tmp * tmp;
    }

    if (current_diff < best_diff) {
      best_diff = current_diff;
      new_center = center;
    }
  }

  if (dev_clusterings[tid] != new_center) {
    *(dev_changes) = true;
    dev_clusterings[tid] = new_center;
  }
}


void warmUpGPU(){
  cudaDeviceSynchronize();
  return;
}

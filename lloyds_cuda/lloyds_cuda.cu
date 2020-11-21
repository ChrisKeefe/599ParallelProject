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
__global__ void reassign(int *dev_num_rows, int *dev_num_cols, int *dev_clusterings, double *dev_cluster_means,
                         double *dev_data_matrix, int *dev_elements_per_cluster);

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

        vector_copy(centers + i * num_cols, data_matrix + center_indices[i] * num_cols, num_cols);
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
  double cluster_means[num_cols * K];
  int elements_per_cluster[K];
  bool changes;

  double *dev_data_matrix;
  double *dev_centers;
  double *dev_cluster_means;
  int *dev_elements_per_cluster;
  int *dev_clusterings;
  bool *dev_changes;
  int *dev_num_rows;
  int *dev_num_cols;
  int *dev_K;

  const unsigned int totalBlocks = ceil(num_rows * 1.0 / BLOCKSIZE);

  warmUpGPU();

  double kernelstart;
  double kerneltime = 0;
  double transfertime = 0;
  double tcpustart = 0;
  double cputime;

  cudaError_t errCode = cudaSuccess;
  
  double tstart = omp_get_wtime();
  double ttransferstart = omp_get_wtime();
  errCode = cudaMalloc(&dev_data_matrix, sizeof(double) * num_rows * num_cols);
  if (errCode != cudaSuccess) {
    cout << "\nError: data_matrix alloc error with code " << errCode << endl;
  }

  errCode = cudaMemcpy(dev_data_matrix, data_matrix, sizeof(double) * num_rows * num_cols, cudaMemcpyHostToDevice);
  if (errCode != cudaSuccess) {
    cout << "\nError: data_matrix memcpy error with code " << errCode << endl;
  }

  errCode = cudaMalloc(&dev_elements_per_cluster, sizeof(int) * K);
  if (errCode != cudaSuccess) {
    cout << "\nError: elements per cluster alloc error with code " << errCode << endl;
  }

  errCode = cudaMalloc(&dev_cluster_means, sizeof(double) * num_cols * K);
  if (errCode != cudaSuccess) {
    cout << "\nError: cluster means alloc error with code " << errCode << endl;
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
  cout << "Initial transfer time: " << transfertime << " seconds" << endl;

  while (1) {
    // Assign points to cluster centers
    changes = false;

    ttransferstart = omp_get_wtime();
    errCode = cudaMemcpy(dev_changes, &changes, sizeof(bool), cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess) {
      cout << "\nError: changes memcpy error with code " << errCode << endl;
    }

    errCode = cudaMemcpy(dev_centers, centers, sizeof(double) * K * num_cols, cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess) {
      cout << "\nError: centers memcpy error with code " << errCode << endl;
    }
    transfertime += omp_get_wtime() - ttransferstart;

    kernelstart = omp_get_wtime();
    lloyds<<<totalBlocks, BLOCKSIZE>>>(dev_data_matrix, dev_centers, dev_clusterings, dev_changes, dev_num_rows, dev_num_cols, dev_K);
    cudaDeviceSynchronize();
    kerneltime += omp_get_wtime() - kernelstart;

    //copy data from device to host
    ttransferstart = omp_get_wtime();
    errCode = cudaMemcpy(&changes, dev_changes, sizeof(bool), cudaMemcpyDeviceToHost);
    if (errCode != cudaSuccess) {
      cout << "\nError: getting changes result from GPU error with code " << errCode << endl;
    }
    transfertime += omp_get_wtime() - ttransferstart;

    // If we didn't change any cluster assignments, we've reached convergence
    if (!changes) {
      break;
    }

    num_iterations++;

    // Find cluster means and reassign centers
    kernelstart = omp_get_wtime();
    errCode = cudaMemset(dev_elements_per_cluster, 0, K * sizeof(int));
    if (errCode != cudaSuccess) {
      cout << "\nError: memsetting elements per cluster error with code " << errCode << endl;
    }

    errCode = cudaMemset(dev_cluster_means, 0, num_cols * K * sizeof(double));
    if (errCode != cudaSuccess) {
      cout << "\nError: memsetting cluster means error with code " << errCode << endl;
    }

    reassign<<<totalBlocks, BLOCKSIZE>>>(dev_num_rows, dev_num_cols, dev_clusterings, dev_cluster_means, dev_data_matrix, dev_elements_per_cluster);
    cudaDeviceSynchronize();
    kerneltime += omp_get_wtime() - kernelstart;

    ttransferstart = omp_get_wtime();
    errCode = cudaMemcpy(elements_per_cluster, dev_elements_per_cluster, sizeof(int) * K, cudaMemcpyDeviceToHost);
    if (errCode != cudaSuccess) {
      cout << "\nError: getting elements per cluster from GPU error with code " << errCode << endl;
    }

    errCode = cudaMemcpy(cluster_means, dev_cluster_means, sizeof(double) * num_cols * K, cudaMemcpyDeviceToHost);
    if (errCode != cudaSuccess) {
      cout << "\nError: getting cluster means from GPU error with code " << errCode << endl;
    }
    transfertime += omp_get_wtime() - ttransferstart;

    tcpustart = omp_get_wtime();
    for (int i = 0; i < K; i++) {
      vector_elementwise_avg(cluster_means + i * num_cols, cluster_means + i * num_cols, elements_per_cluster[i], num_cols);
    }

    // Replace the old cluster means with the new using only three assignments.
    double *temp = centers;
    centers = cluster_means;
    cluster_means = temp;

    cputime += omp_get_wtime() - tcpustart;
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
  printf("Time taken for %d clusters: %f seconds\nkernel: %f seconds"
         "\ntotaltransfer: %f seconds\nCPU time: %f seconds\n\n",
         K, tend - tstart, kerneltime, transfertime, cputime);

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
      tmp = dev_data_matrix[tid * *dev_num_cols + col] - dev_centers[center * *dev_num_cols + col];
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


__global__ void reassign(int *dev_num_rows, int *dev_num_cols, int *dev_clusterings, double *dev_cluster_means,
                         double *dev_data_matrix, int *dev_elements_per_cluster) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= *dev_num_rows) {
    return;
  }

  unsigned int cluster = dev_clusterings[tid];

  for (unsigned int i = 0; i < *dev_num_cols; i++) {
    atomicAdd(&dev_cluster_means[cluster * *dev_num_cols + i], dev_data_matrix[tid * *dev_num_cols + i]);
  }

  atomicAdd(&dev_elements_per_cluster[cluster], int(1));
}


void warmUpGPU(){
  cudaDeviceSynchronize();
  return;
}

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

__global__ void adjust_bounds( double *dev_u_bounds, double *dev_l_bounds, double *dev_centers, 
                               double *dev_prev_centers, int *dev_clustering, double *dev_drifts,
                               int *dev_num_rows, int *dev_num_cols, int *dev_K){
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
  double *prev_centers = (double *)malloc(K * num_cols * sizeof(double));
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
  
  // Create vars and allocate data for GPU
  const unsigned int totalBlocks = ceil(num_rows * 1.0 / BLOCKSIZE);
  warmUpGPU();

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


  double kernel_start;
  double kernel_time = 0;
  double transfer_time = 0;
  double t_cpu_start = 0;
  double cpu_time;

  cudaError_t errCode = cudaSuccess;
  
  double t_start = omp_get_wtime();
  double t_transfer_start = t_start;

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

  transfer_time += omp_get_wtime() - t_transfer_start;
  cout << "Initial transfer time: " << transfer_time << " seconds" << endl;

  // #########################
  // # BEGIN ELKAN MAIN LOOP #
  // #########################
  
  // TODO: I suspect we're going to need additional memory allocations: 
  // u_bound, l_bound, s, z, drifts, ctr_ctr_dists, prev_clousterings, bound_not_tight?

  while (1) {
    changes = false;
    // send changes flag to GPU and time the transfer
    t_transfer_start = omp_get_wtime();
    errCode = cudaMemcpy(dev_changes, &changes, sizeof(bool), cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess) {
    cout << "\nError: changes memcpy error with code " << errCode << endl;
    transfer_time += omp_get_wtime() - t_transfer_start;

    // ###############################################################################
    // Calculate center-center distances with OpenMP (K>=64 uncommon, xfer too costly)
    // ###############################################################################
    #pragma omp parallel for private (i, j, tmp_diff, min_diff) \
        shared(ctr_ctr_dists, centers, num_cols)
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

    // #################################
    // Assign points to cluster centers
    // #################################
    // TODO: transfer data, implement and run assign_points kernel, time


    // ######################################################################
    // If we didn't change any cluster assignments, we've reached convergence
    // ######################################################################
    errCode = cudaMemcpy(&changes, dev_changes, sizeof(bool), cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess) {
      cout << "\nError: changes memcpy error with code " << errCode << endl;
    }

    if (!changes) {
      break;
    }

    num_iterations++;

    // TODO: is this lil guy in the right place?
    // Capture current centers for later re-use
    memcpy(prev_centers, centers, num_cols * K)

    // #######################################
    // Find cluster means and reassign centers
    // #######################################
    errCode = cudaMemset(dev_elements_per_cluster, 0, K * sizeof(int));
    if (errCode != cudaSuccess) {
      cout << "\nError: memsetting elements per cluster error with code " << errCode << endl;
    }

    errCode = cudaMemset(dev_cluster_means, 0, num_cols * K * sizeof(double));
    if (errCode != cudaSuccess) {
      cout << "\nError: memsetting cluster means error with code " << errCode << endl;
    }

    kernel_start = omp_get_wtime();
    reassign<<<totalBlocks, BLOCKSIZE>>>(dev_num_rows, dev_num_cols, dev_clusterings, dev_cluster_means, dev_data_matrix, dev_elements_per_cluster);
    cudaDeviceSynchronize();
    kernel_time += omp_get_wtime() - kernel_start;

    t_transfer_start = omp_get_wtime();
    errCode = cudaMemcpy(elements_per_cluster, dev_elements_per_cluster, sizeof(int) * K, cudaMemcpyDeviceToHost);
    if (errCode != cudaSuccess) {
      cout << "\nError: getting elements per cluster from GPU error with code " << errCode << endl;
    }

    errCode = cudaMemcpy(cluster_means, dev_cluster_means, sizeof(double) * num_cols * K, cudaMemcpyDeviceToHost);
    if (errCode != cudaSuccess) {
      cout << "\nError: getting cluster means from GPU error with code " << errCode << endl;
    }
    transfer_time += omp_get_wtime() - t_transfer_start;

    t_cpu_start = omp_get_wtime();
    for (int i = 0; i < K; i++) {
      vector_elementwise_avg(cluster_means + i * num_cols, cluster_means + i * num_cols, elements_per_cluster[i], num_cols);
    }

    // Replace the old cluster means with the new using only three assignments.
    double *temp = centers;
    centers = cluster_means;
    cluster_means = temp;
    cpu_time += omp_get_wtime() - t_cpu_start;

    // ###########################################
    // Compute centroid drift since last iteration
    // ###########################################
    #pragma omp parallel for private(this_ctr, tmp_diff) \
            shared(centers, prev_centers, num_cols, drifts)
    for (this_ctr = 0; this_ctr < K; this_ctr++) {
      vector_sub(tmp_diff, centers[this_ctr], prev_centers[this_ctr], num_cols);
      drifts[this_ctr] = vector_L2_norm(tmp_diff, num_cols);
    }

    // ###########################################
    // Adjust bounds to account for centroid drift
    // ###########################################
    // TODO: transfer data, call adjust_bounds (below), time
    // TODO: Please double-check the code in this guy!
  }

  //  work on CPU
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

/*
Adjusts the upper and lower bounds to accomodate for centroid drift
*/
__global__ void adjust_bounds( double *dev_u_bounds, double *dev_l_bounds, double *dev_centers, 
                               double *dev_prev_centers, int *dev_clustering, double *dev_drifts,
                               int *dev_num_rows, int *dev_num_cols, int *dev_K){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= *dev_num_rows) return;

    double tmp_diff[*dev_num_cols];
    vector_sub(tmp_diff, dev_centers[dev_clusterings[tid]], dev_prev_centers[dev_clusterings[tid]], *dev_num_cols);
    dev_u_bounds[tid] += vector_L2_norm(tmp_diff, *dev_num_cols);

    for (int this_ctr = 0; this_ctr < *dev_K; this_ctr++) {
      l_bounds[tid * (*dev_K) + this_ctr] -= drifts[this_ctr];
    }
  }
}

/*
Reassigns centroids to their new cluster means
*/
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


/*
Warms up the GPU so that timings are accurate/consistent
*/
void warmUpGPU(){
cudaDeviceSynchronize();
return;
}

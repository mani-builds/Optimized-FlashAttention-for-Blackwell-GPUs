#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Standard Attention Implementaion
// Basic tiled matmul + naive softmax
#define TILE_WIDTH 8
__global__ void matmul_transpose_kernel(float *A, float *B, float *C,int N, int d){

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // each TILE process a block of threads
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Row and Col for the P element
  int outRow = by * TILE_WIDTH + ty;
  int outCol = bx * TILE_WIDTH + tx;


  if (outRow < N && outCol < N){
  // Loop over M and N tiles to computer C
    float sum = 0.0;
    for (int ph=0; ph < d / TILE_WIDTH; ph++){
      // Note we are doing A * B^T
      // Collaborative loading of M and N tiles
      Mds[ty][tx] = A[outRow*d + ph*TILE_WIDTH + tx];
      Nds[ty][tx] = B[outCol*d + ph*TILE_WIDTH + ty];
      __syncthreads();

      for (int i=0; i<TILE_WIDTH; i++){
        sum += Mds[ty][i] * Nds[tx][i];
      }
      __syncthreads();
    }
    C[outRow * N + outCol] = sum;
  }
}

__global__ void matmul_kernel(float *A, float *B, float *C, int N, int d, int width){

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int outRow = by * TILE_WIDTH + ty;
  int outCol = bx * TILE_WIDTH + tx;

  if (outRow < N && outCol < d){
  float sum = 0.0;
  for (int ph = 0; ph < width / TILE_WIDTH; ph++) {
    Mds[ty][tx] = A[outRow*N + ph*TILE_WIDTH + tx];
    Nds[ty][tx] = B[(ph*TILE_WIDTH + ty) * d + outCol];
    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; i++) {
      sum += Mds[ty][i] * Nds[i][tx];
    }
    __syncthreads();
  }
  C[outRow * width + outCol] = sum;
}
}

__global__ void max_rowwise(float *A, float *output, int row_size, int width) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  // float max_value = 0.0f;
  if (row < row_size) {
    for (int i = 0; i < width; i++) {
      if (A[row*width + i] > output[row]) { output[row] = A[row*width + i];}
    }
  }

}
__global__ void numerator_softmax(float *A, float *row_max, float *numerator, int N, int width){
  // Row wise calculation
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < N && col < N) {
  numerator[row*width + col] = exp(A[row * width + col] - row_max[row]);
  }
}

__global__ void denominator_softmax(float *numa, float *den, int N, int width) {
  // Row wise calculation
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < N) {
    float sum = 0.0f;
    for (int i = 0; i < width; i++) {
      sum += numa[row * width + i];
    }
    den[row] = sum;
    }
}


int main(){
  float *Q_h;
  float *K_h;
  float *V_h;

  int N = 1024; // No. of sequences
  int d = 32; // Head dimensions

  Q_h = (float *) malloc(N * d * sizeof(float));
  K_h = (float *) malloc(N * d * sizeof(float));
  V_h = (float *) malloc(N * d * sizeof(float));

  int rand_max = 10;
  srand(22);

  // Generate random numbers
  for(int i = 0; i < N * d; i++) {
    Q_h[i] = rand() % rand_max + 1;
    K_h[i] = rand() % rand_max + 1;
    V_h[i] = rand() % rand_max + 1;
  }

  printf("\nFirst 100 values of Q: \n");
  for(int i = 0; i < 100; i++) {
    printf("%f \t", Q_h[i]);
    // printf("%f \t", K_h[i]);
    // printf("%f \t", V_h[i]);
  }
  printf("\nFirst N values of K: \n");
  for(int i = 0; i < 100; i++) {
    printf("%f \t", K_h[i]);
    // printf("%f \t", K_h[i]);
    // printf("%f \t", V_h[i]);
  }
  printf("\nFirst N values of V: \n");
  for(int i = 0; i < 100; i++) {
    printf("%f \t", V_h[i]);
    // printf("%f \t", K_h[i]);
    // printf("%f \t", V_h[i]);
  }

  float *Q;
  float *K;
  float *V;

  cudaMalloc(&Q, N * d * sizeof(float));
  cudaMalloc(&K, N * d * sizeof(float));
  cudaMalloc(&V, N * d * sizeof(float));

  cudaMemcpy(Q, Q_h, N * d * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(K, K_h, N * d * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(V, V_h, N * d * sizeof(float), cudaMemcpyHostToDevice);

  // matmul of Q and K^T i.e S = Q*K^T
  float *S_h;
  float *S;
  S_h = (float *) malloc(N * N * sizeof(float));
  cudaMalloc(&S, N * N * sizeof(float));

  dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
  dim3 blocksPerGrid((N * threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N * threadsPerBlock.y - 1) / threadsPerBlock.y);
  matmul_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(Q, K, S, N, d);

  cudaMemcpy(S_h, S, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("\nFirst N values of S: \n");
  for(int i =0; i<100; i++){
    printf("%f \t", S_h[i]);
  }

  // Max row
  float *max_row_h;
  float *max_row;
  max_row_h = (float *) malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    max_row_h[i] = 0.0f;
  }
  cudaMalloc(&max_row, N * sizeof(float));
  cudaMemcpy(max_row, max_row_h, N * sizeof(float), cudaMemcpyHostToDevice);

  max_rowwise<<<(N + 32 -1)/32, 32>>>(S, max_row, N, N);
  cudaMemcpy(max_row_h, max_row, N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("\nN values of max_row: \n");
  for(int i =0; i<100; i++){
    printf("%f \t", max_row_h[i]);
  }

  // Numerator: e^x - m(x)
  float *numerator_h;
  float *numerator;
  numerator_h = (float *) malloc(N * N * sizeof(float));
  cudaMalloc(&numerator, N * N * sizeof(float));

  dim3 threadsNum(16, 16);
  dim3 blocksNum((N + threadsNum.x - 1) / threadsNum.x,
                 (N + threadsNum.y -1) / threadsNum.y);
  numerator_softmax<<<blocksNum,threadsNum>>>(S, max_row, numerator, N, N);
  cudaMemcpy(numerator_h, numerator, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("\nFirst N values of numerator: \n");
  for(int i =0; i<100; i++){
    printf("%f \t", numerator_h[i]);
  }

  // P = Softmax(numerator/denominator)
  float *P_h;
  float *P;
  P_h = (float *) malloc(N*N*sizeof(float));
  cudaMalloc(&P, N*N*sizeof(float));

  float *den_h;
  float *den;
  den_h = (float *) malloc(N*sizeof(float));
  cudaMalloc(&den, N*sizeof(float));

  denominator_softmax<<<(d + 32 -1)/32, 32>>>(numerator, den, N, N);
  cudaMemcpy(den_h, den, N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("\nN values of denominator: \n");
  for(int i =0; i<100; i++){
    printf("%f \t", den_h[i]);
  }

  for (int i = 0; i < N; i++) {
    for (int j=0; j < N; j++){
    P_h[i*N + j] = numerator_h[i*N + j] / den_h[i];
    }
  }

  printf("\nFirst N values of softmax: \n");
  for(int i =0; i<100; i++){
    printf("%f \t", P_h[i]);
  }

  cudaMemcpy(P, P_h, N*N*sizeof(float), cudaMemcpyHostToDevice);

  // matmul of P and V i.e O = P*V
  float *O_h;
  float *O;
  O_h = (float *) malloc(N * d * sizeof(float));
  cudaMalloc(&O, N * d * sizeof(float));
  dim3 threads(TILE_WIDTH,TILE_WIDTH);
  dim3 blocks((d + threads.x - 1) / threads.x,
              (N + threads.y - 1) / threads.y);

  matmul_kernel<<<blocks, threads>>>(O, V, O, N, d, d);

  cudaMemcpy(O_h, O, N * d * sizeof(float), cudaMemcpyDeviceToHost);

  printf("\nFirst N values of O: \n");
  for(int i =0; i<100; i++){
    printf("%f \t", O_h[i]);
  }


  free(Q_h);
  free(K_h);
  free(V_h);
  free(max_row_h);
  free(numerator_h);
  free(S_h);
  free(den_h);
  free(O_h);
  cudaFree(den);
  cudaFree(O);
  cudaFree(numerator);
  cudaFree(S);
  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  return 0;
}

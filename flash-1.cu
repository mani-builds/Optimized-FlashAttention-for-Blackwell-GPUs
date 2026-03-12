#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>

#define TILE_WIDTH 8

__global__ void forward_kernel(float *Q, float *K, float *V, int N, int d,
                               int Br, int Bc,
                               float *O, float *l, float *m) {
  // Thread mapping working differently from std matmul kernle
  int tx = threadIdx.x;

  // Each Block handles one Br x d of Oi(output) tile.Each block loads
  // a tile Qi and then iterates through all tiles of Kj and Vj
  //
  //
  // Each thread is organized to maximize memory coalescing and row-wise ops.
  // Each thread is assigned to one or more row within the Br block
  // Each thread calculates dot product of QiKi for all Bc cols
  // The same thread is resposible for "Online" part of softmax

  extern __shared__ float sram[];
  // Partition the shared memory (SRAM)
  // [Qi (Brxd) | Kj (Bcxd) | Vj (Bcxd) | Oi (Brxd) | li (Br) | mi (Br)]
  float *Qi = sram;
  float *Kj = Qi + (Br * d);
  float *Vj = Kj + (Bc * d);
  float *Oi = Vj + (Bc * d);
  float *li = Oi + (Br * d);
  float *mi = li + Br;

  // Line 5: Loop over blocks of K, V (j = 1 to Tc)
  for (int j = 0; j < N / Bc; j++) {
    // Line 6: Load Kj and Vj
    if (tx < Bc * d){
    Kj[tx] = K[j*Bc*d + tx];
    Vj[tx] = V[j*Bc*d + tx];
    }
    __syncthreads();
    // Line 7: Loop over blocks of Q (i = 1 to Tr)

    for (int i = 0; i < N / Br; i++) {
      // Line 8: Load Qi, Oi, li, mi
      if (tx < Br){
        for (int k = 0; k < d; k++) {
          Qi[tx * d + k] = Q[(i * Br + tx)*d + k];
          Oi[tx * d + k] = O[(i * Br + tx)*d + k];
        }
        li[tx] = l[i * Br + tx];
        mi[tx] = m[i * Br + tx];
      }
      __syncthreads();
      // Line 9 & 10: Compute Sij, mij_hat, Pij_hat, lij_hat
      if(tx < Br){
        float row_m_prev = mi[tx];
        float row_l_prev = li[tx];
        // Compute max for the current horizontal tile (Sij)
        float row_m_curr = -INFINITY;
        for (int col = 0; col < Bc; col++) {
          float score = 0.0;
          for (int k = 0; k < d; k++) {
            score += Qi[tx * d + k] * Kj[col * d + k];
          }
          // scale score here if needed (1/sqrt(d))
          row_m_curr = fmaxf(row_m_curr, score);
        }
        // Compute sum of exp for current tile
        float row_l_curr = 0;
        for (int col = 0; col < Bc; col++) {
          float score = 0.0;
          for (int k = 0; k < d; k++) {
            score += Qi[tx * d + k] * Kj[col * d + k];
          }
          // scale score here if needed (1/sqrt(d))
          row_l_curr += expf(score - row_m_curr);
        }
        // Line 11: Keep track of running sum and max
        float row_m_new = fmaxf(row_m_prev, row_m_curr);
        float row_l_new = expf(row_m_prev - row_m_new) * row_l_prev +
                          expf(row_m_curr - row_m_new) * row_l_curr;
        //Line 12: Rescale Oi and add new contribution
        for (int k = 0; k < d; k++) {
            float pv_sum = 0;
            for (int col = 0; col < Bc; col++) {
                float score = 0;
                for (int kk = 0; kk < d; kk++) score += Qi[tx * d + kk] * Kj[col * d + kk];
                pv_sum += expf(score - row_m_curr) * Vj[col * d + k];
            }
            float o_val = (row_l_prev * expf(row_m_prev - row_m_new) * Oi[tx * d + k] +
                                   expf(row_m_curr - row_m_new) * pv_sum) / row_l_new;
            // Line 12 / 13 : Write back to HBM
            O[(i * Br + tx)*d + k] = o_val;
        }
        l[i * Br + tx] = row_l_new;
        m[i * Br + tx] = row_m_new;
      }
      __syncthreads();
    }
  }
 }

int main() {
  float *Q_h, *K_h, *V_h;

  int N = 32;
  int d = 8;

  Q_h = (float *) malloc(N * d * sizeof(float));
  K_h = (float *) malloc(N * d * sizeof(float));
  V_h = (float *) malloc(N * d * sizeof(float));

  float *Q, *K, *V;
  cudaMalloc(&Q, N*d*sizeof(float));
  cudaMalloc(&K, N*d*sizeof(float));
  cudaMalloc(&V, N*d*sizeof(float));

  cudaMemcpy(Q, Q_h, N*d*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(K, K_h, N*d*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(V, V_h, N*d*sizeof(float), cudaMemcpyHostToDevice);

  int rnd_max = 100;
  srand(22);

  for (int i = 0; i < N * d; i++) {
    Q_h[i] = rand() % rnd_max - 1;
    K_h[i] = rand() % rnd_max - 1;
    V_h[i] = rand() % rnd_max - 1;
  }

  // SRAM size
  int M = 16;
  // Setting block row and col
  int Bc = ceil(M / 4*d);
  int Br = MIN(ceil(M/4*d),d);

  // Initialize in HBM
  float *O_h, *l_h, *m_h;

  O_h = (float *) malloc(N * d * sizeof(float));
  l_h = (float *)malloc(N * sizeof(float));
  m_h = (float *) malloc(N * sizeof(float));

  for (int i = 0; i < N * d; i++) {
    O_h[i] = 0.0;
  }
  for (int i = 0; i < N; i++) {
    l_h[i] = 0.0;
  }
  for (int i = 0; i < N ; i++) {
    m_h[i] = -INFINITY;
  }

  float *O, *l, *m;
  cudaMalloc(&O, N * d * sizeof(float));
  cudaMalloc(&l, N * sizeof(float));
  cudaMalloc(&m, N*sizeof(float));

  cudaMemcpy(O, O_h, N*d*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(l, l_h, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(m, m_h, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N + threadsPerBlock.y -1) / threadsPerBlock.y);


  free(O_h);
  free(Q_h);
  free(K_h);
  free(V_h);
  free(l_h);
  free(m_h);
  cudaFree(O);
  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  cudaFree(l);
  cudaFree(m);

  return 0;
}

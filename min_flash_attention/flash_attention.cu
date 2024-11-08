// resources:
// - https://github.com/tspeterkim/flash-attention-minimal
// - https://github.com/leloykun/flash-hyperbolic-attention-minimal
// - https://github.com/karpathy/llm.c

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <torch/types.h>

// #define ceil_div(x, y) (ceil((float)(x) / y))
#define ceil_div(x, y) ((x + y - 1) / y)
// a basic reimplementation of flash attention v1 paper.
// Note: I'm new to CUDA, so might not be 100% accurate
//

#define TILE_DIM 32 // Tile dimension for efficient memory usage

// THIS IS THE FORWARD PASS FOR FLASH ATTENTION V1 KERNEL
__global__ void forward2_kernel(const float *Q, const float *K, const float *V,
                                const int N, const int d, const int Tc,
                                const int Tr, const int Bc, const int Br,
                                const float softmax_scale, float *l, float *m,
                                float *O) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y; // batch and head index

  // Offset into Q,K,V,O - different for each batch and head
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
  int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for L

  // Define SRAM for Q,K,V,S
  extern __shared__ float sram[];
  int tile_size = Bc * d; // size of Qi, Kj, Vj
  float *Qi = sram;
  float *Kj = &sram[tile_size];
  float *Vj = &sram[tile_size * 2];
  float *S = &sram[tile_size * 3];

  for (int i = 0; i < Tr; ++i) {
    if (i * Br + tx >= N)
      break; // break if we are done with the sequence

    // Load Qi from HBM to SRAM, l and m to registers
    for (int x = 0; x < d; x++) {
      Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
    }
    float row_m_prev = -INFINITY;
    float row_l_prev = 0;

    // Causal mask: j <= i
    for (int j = 0; j <= i; ++j) {
      __syncthreads();
      // Load Kj, Vj from HBM to SRAM
      for (int x = 0; x < d; x++) {
        Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
        Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
      }
      __syncthreads();
      // S_i^j = softmax_scale * QiKj^T
      // S_i^j[tx][y] = softmax_scale * Sum_{x = 0}^{d-1} Qi[tx][x] * Kj[y][x]
      float row_m = -INFINITY;
      for (int y = 0; y < Bc; y++) {
        if (j * Bc + y >= N)
          break; // break if we are done with the sequence
        if (i * Br + tx < j * Bc + y)
          break;
        float sum = 0;
        for (int x = 0; x < d; x++)
          sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
        sum *= softmax_scale;
        S[(Bc * tx) + y] = sum;

        if (sum > row_m)
          row_m = sum;
      }

      // m_i^j = max(m_i^j-1, row_max(S_i^j))
      float new_row_m = max(row_m_prev, row_m);

      // P_i^j = exp(S_i^j - m_i^j)
      // P_i^j[tx][y] = exp(S_i^j[tx][y] - m_i^j)
      float row_l = 0;
      for (int y = 0; y < Bc; y++) {
        if (j * Bc + y >= N)
          break; // break if we are done with the sequence
        if (i * Br + tx < j * Bc + y)
          break;
        S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - new_row_m);
        row_l += S[(Bc * tx) + y];
      }

      // l_i^j = (exp(m_i^j-1 - m_i^j) * l_i^j-1) + row_sum(P_i^j)
      float row_m_exp = __expf(row_m_prev - new_row_m);
      float new_row_l = (row_m_exp * row_l_prev) + row_l;

      // O_i^j = diag(exp(m_i^j-1 - m_i^j))^-1 * O_i^j-1 + P_i^jVj
      for (int x = 0; x < d; x++) {
        float pv = 0; // Pij * Vj
        for (int y = 0; y < Bc; y++) {
          if (j * Bc + y >= N)
            break; // break if we are done with the sequence
          if (i * Br + tx < j * Bc + y)
            break;
          pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
        }
        O[qkv_offset + (tile_size * i) + (tx * d) + x] =
            row_m_exp * O[qkv_offset + (tile_size * i) + (tx * d) + x] + pv;
      }

      // Update m and l
      row_m_prev = new_row_m;
      row_l_prev = new_row_l;
    }

    // O_i = diag(l_i^{Tc})^-1 * O_i^{Tc}
    for (int x = 0; x < d; x++)
      O[qkv_offset + (tile_size * i) + (tx * d) + x] /= row_l_prev;
    // L_i = m_i^{Tc} + log(l_i^{Tc})
    l[lm_offset + (Br * i) + tx] = row_m_prev + __logf(row_l_prev);
  }
}

__global__ void backward2_kernel(const float *Q, const float *K, const float *V,
                                 const float *O, const float *dO,
                                 const float *L, const int N, const int d,
                                 const int Tc, const int Tr, const int Bc,
                                 const int Br, const float softmax_scale,
                                 float *dQ, float *dK, float *dV) {
  // TODO: need to do this!!
}

// THIS IS THE FORWARD PASS FOR FLASH ATTENTION V1 KERNEL
__global__ void forward1_kernel(const float *Q, const float *K, const float *V,
                                const int N, const int d, const int Tc,
                                const int Tr, const int Bc, const int Br,
                                const float softmax_scale, float *l, float *m,
                                float *O) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y; // batch and head index

  // Offset into Q,K,V,O,l,m - different for each batch and head
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
  int lm_offset = (bx * gridDim.y * N) + (by * N); // offset for l and m

  // Define SRAM for Q,K,V,S
  extern __shared__ float sram[];
  int tile_size = Bc * d; // size of Qi, Kj, Vj
  float *Qi = sram;
  float *Kj = &sram[tile_size];
  float *Vj = &sram[tile_size * 2];
  float *S = &sram[tile_size * 3];

  for (int j = 0; j < Tc; j++) {

    // Load Kj, Vj to SRAM
    for (int x = 0; x < d; x++) {
      Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
      Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
    }
    __syncthreads(); // such that the inner loop can use the correct Kj, Vj

    for (int i = 0; i < Tr; i++) {

      // Load Qi to SRAM, l and m to registers
      for (int x = 0; x < d; x++) {
        Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
      }
      float row_m_prev = m[lm_offset + (Br * i) + tx];
      float row_l_prev = l[lm_offset + (Br * i) + tx];

      // S = QK^T, row_m = rowmax(S)
      float row_m = -INFINITY;
      for (int y = 0; y < Bc; y++) {
        float sum = 0;
        for (int x = 0; x < d; x++) {
          sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
        }
        sum *= softmax_scale;
        S[(Bc * tx) + y] = sum;

        if (sum > row_m)
          row_m = sum;
      }

      // P = exp(S - row_m), row_l = rowsum(P)
      float row_l = 0;
      for (int y = 0; y < Bc; y++) {
        S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
        row_l += S[(Bc * tx) + y];
      }

      // THIS IS THE FORWARD PASS FOR FLASH ATTENTION V1 KERNEL
      __global__ void forward2_kernel(
          const float *Q, const float *K, const float *V, const int N,
          const int d, const int Tc, const int Tr, const int Bc, const int Br,
          const float softmax_scale, float *l, float *m, float *O) {
        int tx = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y; // batch and head index

        // Offset into Q,K,V,O - different for each batch and head
        int qkv_offset =
            (bx * gridDim.y * N * d) + (by * N * d);     // gridDim.y = nh
        int lm_offset = (bx * gridDim.y * N) + (by * N); // offset for L

        // Define SRAM for Q,K,V,S
        extern __shared__ float sram[];
        int tile_size = Bc * d; // size of Qi, Kj, Vj
        float *Qi = sram;
        float *Kj = &sram[tile_size];
        float *Vj = &sram[tile_size * 2];
        float *S = &sram[tile_size * 3];

        for (int i = 0; i < Tr; ++i) {
          if (i * Br + tx >= N)
            break; // break if we are done with the sequence

          // Load Qi from HBM to SRAM, l and m to registers
          for (int x = 0; x < d; x++) {
            Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
          }
          float row_m_prev = -INFINITY;
          float row_l_prev = 0;

          // Causal mask: j <= i
          for (int j = 0; j <= i; ++j) {
            __syncthreads();
            // Load Kj, Vj from HBM to SRAM
            for (int x = 0; x < d; x++) {
              Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
              Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }
            __syncthreads();
            // S_i^j = softmax_scale * QiKj^T
            // S_i^j[tx][y] = softmax_scale * Sum_{x = 0}^{d-1} Qi[tx][x] *
            // Kj[y][x]
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
              if (j * Bc + y >= N)
                break; // break if we are done with the sequence
              if (i * Br + tx < j * Bc + y)
                break;
              float sum = 0;
              for (int x = 0; x < d; x++)
                sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
              sum *= softmax_scale;
              S[(Bc * tx) + y] = sum;

              if (sum > row_m)
                row_m = sum;
            }

            // m_i^j = max(m_i^j-1, row_max(S_i^j))
            float new_row_m = max(row_m_prev, row_m);

            // P_i^j = exp(S_i^j - m_i^j)
            // P_i^j[tx][y] = exp(S_i^j[tx][y] - m_i^j)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
              if (j * Bc + y >= N)
                break; // break if we are done with the sequence
              if (i * Br + tx < j * Bc + y)
                break;
              S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - new_row_m);
              row_l += S[(Bc * tx) + y];
            }

            // l_i^j = (exp(m_i^j-1 - m_i^j) * l_i^j-1) + row_sum(P_i^j)
            float row_m_exp = __expf(row_m_prev - new_row_m);
            float new_row_l = (row_m_exp * row_l_prev) + row_l;

            // O_i^j = diag(exp(m_i^j-1 - m_i^j))^-1 * O_i^j-1 + P_i^jVj
            for (int x = 0; x < d; x++) {
              float pv = 0; // Pij * Vj
              for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N)
                  break; // break if we are done with the sequence
                if (i * Br + tx < j * Bc + y)
                  break;
                pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
              }
              O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                  row_m_exp * O[qkv_offset + (tile_size * i) + (tx * d) + x] +
                  pv;
            }

            // Update m and l
            row_m_prev = new_row_m;
            row_l_prev = new_row_l;
          }

          // O_i = diag(l_i^{Tc})^-1 * O_i^{Tc}
          for (int x = 0; x < d; x++)
            O[qkv_offset + (tile_size * i) + (tx * d) + x] /= row_l_prev;
          // L_i = m_i^{Tc} + log(l_i^{Tc})
          l[lm_offset + (Br * i) + tx] = row_m_prev + __logf(row_l_prev);
        }
      }

      // Compute new m and l
      float row_m_new = max(row_m_prev, row_m);
      float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                        (__expf(row_m - row_m_new) * row_l);

      // Write O, l, m to HBM
      for (int x = 0; x < d; x++) {
        float pv = 0; // Pij * Vj
        for (int y = 0; y < Bc; y++) {
          pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
        }
        O[qkv_offset + (tile_size * i) + (tx * d) + x] =
            (1 / row_l_new) *
            ((row_l_prev * __expf(row_m_prev - row_m_new) *
              O[qkv_offset + (tile_size * i) + (tx * d) + x]) +
             (__expf(row_m - row_m_new) * pv));
      }
      m[lm_offset + (Br * i) + tx] = row_m_new;
      l[lm_offset + (Br * i) + tx] = row_l_new;
    }
    __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop
  }
}

// THIS IS THE BACKWARD PASS FOR THE FLASHATTENTION V1 KERNEL.
// NOTE: FlashAttention’s main performance gains are seen during the forward
// pass, particularly in inference. While this backward pass is required for
// training, it may not benefit as much from FlashAttention's optimizations.
__global__ void backward1_kernel(const float *Q, const float *K, const float *V,
                                 const float *O, const float *dO,
                                 const float *l, const float *m, const int N,
                                 const int d, const int Tc, const int Tr,
                                 const int Bc, const int Br,
                                 const float softmax_scale, float *dQ,
                                 float *dK, float *dV) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y; // batch and head index

  // Offset into Q,K,V,O,l,m - different for each batch and head
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
  int lm_offset = (bx * gridDim.y * N) + (by * N); // offset for l and m

  // Define SRAM for Q,K,V,S
  extern __shared__ float sram[];
  int col_tile_size = Bc * d; // size of Kj, Vj
  int row_tile_size = Br * d; // size of Qi
  float *Kj = sram;
  float *Vj = &sram[col_tile_size];

  float *dKj = &sram[col_tile_size * 2];
  float *dVj = &sram[col_tile_size * 3];

  float *Qi = &sram[col_tile_size * 4];
  float *Oi = &sram[col_tile_size * 4 + row_tile_size];
  float *dOi = &sram[col_tile_size * 4 + row_tile_size * 2];

  // We also use S for P. Likewise, we use dS for dP.
  // We can reuse the same memory because we don't need S and P at the same
  // time. We also don't need dS and dP at the same time.
  float *S = &sram[col_tile_size * 4 + row_tile_size * 3];
  float *dS = &sram[col_tile_size * 4 + row_tile_size * 3 + Bc * Br];

  for (int j = 0; j < Tc; j++) {

    // Load Kj, Vj to SRAM
    for (int x = 0; x < d; x++) {
      Kj[(tx * d) + x] = K[qkv_offset + (col_tile_size * j) + (tx * d) + x];
      Vj[(tx * d) + x] = V[qkv_offset + (col_tile_size * j) + (tx * d) + x];
    }

    // Initialize dKj, dVj to 0
    for (int x = 0; x < d; x++) {
      dKj[(tx * d) + x] = 0;
      dVj[(tx * d) + x] = 0;
    }

    for (int i = j; i < Tr; i++) {
      __syncthreads();
      // Load Qi, Oi, dOi, dQi, li, mi to SRAM
      // Also load l, m to registers
      for (int x = 0; x < d; x++) {
        Qi[(tx * d) + x] = Q[qkv_offset + (row_tile_size * i) + (tx * d) + x];
        Oi[(tx * d) + x] = O[qkv_offset + (row_tile_size * i) + (tx * d) + x];
        dOi[(tx * d) + x] = dO[qkv_offset + (row_tile_size * i) + (tx * d) + x];
      }
      float m_curr = m[lm_offset + (Br * i) + tx];
      float l_curr = l[lm_offset + (Br * i) + tx];

      // Sij = softmax_scale * QiKj^T
      // Sij[tx][y] = softmax_scale * Sum_{y = 0}^{Bc-1} Qi[tx][x] * Kj[y][x]
      for (int y = 0; y < Bc; y++) {
        float sum = 0;
        for (int x = 0; x < d; x++) {
          sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
        }
        sum *= softmax_scale;
        if (i * Br + tx < j * Bc + y)
          sum = -INFINITY;
        S[(Bc * tx) + y] = sum;
      }

      // Pij = diag(li)^-1 * exp(Sij - mi)
      // Pij[tx][y] = (1 / li[tx]) * exp(Sij[tx][y] - mi[tx])
      for (int y = 0; y < Bc; y++) {
        if (i * Br + tx < j * Bc + y)
          S[(Bc * tx) + y] = 0;
        else
          S[(Bc * tx) + y] = (1 / l_curr) * __expf(S[(Bc * tx) + y] - m_curr);
      }
      __syncthreads();
      // dVj <- dVj + Pij^T * dOi
      // dVj[tx][x] = dVj[tx][x] + Sum_{y = 0}^{Br-1} Pij[y][tx] * dOi[tx][x]
      for (int x = 0; x < d; x++) {
        float sum = 0;
        for (int y = 0; y < Br; y++) {
          sum += S[(Bc * y) + tx] * dOi[(tx * d) + x];
        }
        atomicAdd(&dVj[(tx * d) + x], sum);
      }

      // dPij <- dOi * Vj^T
      // dPij[tx][y] = Sum_{x = 0}^{d-1} dOi[tx][x] * Vj[y][x]
      for (int y = 0; y < Bc; y++) {
        float sum = 0;
        for (int x = 0; x < d; x++) {
          sum += dOi[(tx * d) + x] * Vj[(y * d) + x];
        }
        dS[(Bc * tx) + y] = sum;
      }

      // Di <- rowsum(dOi * Oi)  (pointwise multiply)
      // Di[tx] = Sum_{x = 0}^{d-1} dOi[tx][x] * Oi[tx][x]
      float Di = 0;
      for (int x = 0; x < d; x++) {
        Di += dOi[(tx * d) + x] * Oi[(tx * d) + x];
      }

      // dSij <- Pij * (dPij - Di)
      // dSij[tx][y] = Pij[tx][y] * (dPij[tx][y] - Di[tx])
      for (int y = 0; y < Bc; ++y) {
        dS[(Bc * tx) + y] = S[(Bc * tx) + y] * (dS[(Bc * tx) + y] - Di);
      }

      // dQi <- dQi + softmax_scale * dSijKj
      // dQ[tx][x] = dQ[tx][x] + softmax_scale * Sum_{y = 0}^{Bc-1} dSij[tx][y]
      // * Kj[y][x]
      for (int x = 0; x < d; x++) {
        float sum = 0;
        for (int y = 0; y < Bc; y++) {
          sum += dS[(Bc * tx) + y] * Kj[(y * d) + x];
        }
        sum *= softmax_scale;
        atomicAdd(&dQ[qkv_offset + (row_tile_size * i) + (tx * d) + x], sum);
      }
      __syncthreads();
      // dKj <- dKj + softmax_scale * dSij^TQi
      // dKj[tx][x] = dKj[tx][x] + softmax_scale * Sum_{y = 0}^{Br-1}
      // dSij[y][tx] * Qi[y][x]
      for (int x = 0; x < d; x++) {
        float sum = 0;
        for (int y = 0; y < Br; y++) {
          sum += dS[(Bc * y) + tx] * Qi[(y * d) + x];
        }
        sum *= softmax_scale;
        atomicAdd(&dKj[(tx * d) + x], sum);
      }
    }

    // Upload Kj, Vj to HRAM
    for (int x = 0; x < d; x++) {
      dK[qkv_offset + (row_tile_size * j) + (tx * d) + x] = dKj[(tx * d) + x];
      dV[qkv_offset + (row_tile_size * j) + (tx * d) + x] = dVj[(tx * d) + x];
    }
  }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      int version) {

  const int B = Q.size(0);                   // batch size
  const int nh = Q.size(1);                  // number of heads
  const int T = Q.size(2);                   // sequence length
  const int C = Q.size(3);                   // number of channels
  const int Tc = ceil_div(T, TILE_DIM);      // number of tiles along columns
  const int Tr = ceil_div(T, TILE_DIM);      // number of tiles along rows
  const float softmax_scale = 1.0 / sqrt(C); // scaling factor in attention

  auto O = torch::zeros_like(Q);     // output tensor
  auto l = torch::zeros({B, nh, N}); // storing softmax computed values
  auto m = torch::full({B, nh, N}, -INFINITY); // stores max values
                                               //
  // SRAM size needed per block, checkout
  // `./quadratic_attention.ipynb` for explanation
  const int sram_size =
      (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Maximum shared memory: %d\nRequested shared memory: %d \n",
         max_sram_size, sram_size);

  dim3 grid_dim(B, nh);     // batch size * number of heads
  dim3 block_dim(TILE_DIM); // Batch threads per tile

  switch (version) {
  case 1:
    forward1_kernel(Q.data_ptr<float>(), K.data_ptr<float>(),
                    V.data_ptr<float>(), T, C, Tc, Tr, TILE_DIM, TILE_DIM,
                    softmax_scale, l.data_ptr<float>(), m.data_ptr<float>(),
                    O.data_ptr<float>());
      );
      break;

  case 2:
    forward2_kernel(Q.data_ptr<float>(), K.data_ptr<float>(),
                    V.data_ptr<float>(), T, C, Tc, Tr, TILE_DIM, TILE_DIM,
                    softmax_scale, l.data_ptr<float>(), m.data_ptr<float>(),
                    O.data_ptr<float>());
      );
      break;
  default:
    printf("Using default flash attention version (v2)");
    forward2_kernel(Q.data_ptr<float>(), K.data_ptr<float>(),
                    V.data_ptr<float>(), T, C, Tc, Tr, TILE_DIM, TILE_DIM,
                    softmax_scale, l.data_ptr<float>(), m.data_ptr<float>(),
                    O.data_ptr<float>());
      );
  }
  return O;
}

std::vector<torch::Tensor> backward(torch::Tensor Q, torch::Tensor K,
                                    torch::Tensor V, torch::Tensor O,
                                    torch::Tensor dO, torch::Tensor l,
                                    torch::Tensor m, int version) {

  const int B = Q.size(0);                   // batch size
  const int nh = Q.size(1);                  // number of heads
  const int T = Q.size(2);                   // sequence length
  const int C = Q.size(3);                   // number of channels
  const int Tc = ceil_div(T, TILE_DIM);      // number of tiles along columns
  const int Tr = ceil_div(T, TILE_DIM);      // number of tiles along rows
  const float softmax_scale = 1.0 / sqrt(C); // scaling factor in attention

  auto dQ = torch::zeros_like(Q); // gradients of query tensor
  auto dK = torch::zeros_like(K); // gradients of key tensor
  auto dV = torch::zeros_like(V); // gradients of value tensor

  // SRAM size needed per block, checkout
  // `./quadratic_attention.ipynb` for explanation
  const int sram_size =
      (4 * col_tile_size * sizeof(float))   // SRAM size for Kj, Vj, dKj, dVj
      + (3 * row_tile_size * sizeof(float)) // SRAM size for Qi, Oi, dOi
      + (2 * Br * Bc * sizeof(float));      // SRAM size for S, dS

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Maximum shared memory: %d\nRequested shared memory: %d \n",
         max_sram_size, sram_size);

  dim3 grid_dim(B, nh);     // batch size * number of heads
  dim3 block_dim(TILE_DIM); // Batch threads per tile

  switch (version) {
  case 1:

    backward1_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), dO.data_ptr<float>(), l.data_ptr<float>(),
        m.data_ptr<float>(), T, C, Tc, Tr, TILE_DIM, TILE_DIM, softmax_scale,
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(), );
  );
  break;

  case 2:
    backward2_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), T, C, Tc,
        Tr, TILE_DIM, TILE_DIM, softmax_scale, l.data_ptr<float>(),
        m.data_ptr<float>(), O.data_ptr<float>());
      );
      break;
  default:
    printf("Using default flash attention version - backward (v1)");

    backward1_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), T, C, Tc,
        Tr, TILE_DIM, TILE_DIM, softmax_scale, l.data_ptr<float>(),
        m.data_ptr<float>(), O.data_ptr<float>());
      );
      break;
  }
  return {dQ, dK, dV};
}

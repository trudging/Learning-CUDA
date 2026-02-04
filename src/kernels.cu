#include <vector>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

#include "../tester/utils.h"

// ===================== TRACE KERNEL =====================
template <typename T>
__global__ void traceKernel(const T* input, T* output, size_t rows, size_t cols) {
  extern __shared__ char shared_mem[];
  T* sdata = reinterpret_cast<T*>(shared_mem);
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t diag_len = min(rows, cols);
  
  // Each thread loads one diagonal element
  T val = T(0);
  if (i < diag_len) {
    val = input[i * cols + i];  // diagonal element at (i, i)
  }
  sdata[tid] = val;
  __syncthreads();
  
  // Parallel reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = sdata[tid] + sdata[tid + s];
    }
    __syncthreads();
  }
  
  // Write result for this block
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}

// Specialization for int atomicAdd (not natively supported on older GPUs)
__device__ inline void atomicAddInt(int* address, int val) {
  atomicAdd(address, val);
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  size_t diag_len = std::min(rows, cols);
  if (diag_len == 0) return T(0);
  
  // Allocate device memory
  T* d_input;
  T* d_output;
  size_t input_size = rows * cols * sizeof(T);
  
  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_output, sizeof(T));
  
  // Copy input to device and initialize output to 0
  cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, sizeof(T));
  
  // Launch kernel
  int blockSize = 256;
  int numBlocks = (diag_len + blockSize - 1) / blockSize;
  size_t sharedMemSize = blockSize * sizeof(T);
  
  traceKernel<T><<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_output, rows, cols);
  
  // Copy result back
  T result;
  cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost);
  
  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  
  return result;
}

// ===================== FLASH ATTENTION KERNEL =====================
// Helper: convert float to type T
template <typename T>
__device__ __forceinline__ T floatToT(float val);

template <>
__device__ __forceinline__ float floatToT<float>(float val) { return val; }

template <>
__device__ __forceinline__ half floatToT<half>(float val) { return __float2half(val); }

// Helper: convert type T to float
template <typename T>
__device__ __forceinline__ float TtoFloat(T val);

template <>
__device__ __forceinline__ float TtoFloat<float>(float val) { return val; }

template <>
__device__ __forceinline__ float TtoFloat<half>(half val) { return __half2float(val); }

template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size, int tgt_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal, float scale) {
  
  // Each thread handles one output element
  // Grid: (batch_size * tgt_seq_len * query_heads, head_dim)
  int linear_idx = blockIdx.x;
  int d = blockIdx.y * blockDim.x + threadIdx.x;
  
  if (d >= head_dim) return;
  
  int b = linear_idx / (tgt_seq_len * query_heads);
  int remainder = linear_idx % (tgt_seq_len * query_heads);
  int t = remainder / query_heads;
  int h = remainder % query_heads;
  
  if (b >= batch_size) return;
  
  // GQA: map query head to kv head
  int kv_h = h / (query_heads / kv_heads);
  
  // Compute attention for this position
  // Q shape: [batch_size, tgt_seq_len, query_heads, head_dim]
  // K shape: [batch_size, src_seq_len, kv_heads, head_dim]
  // V shape: [batch_size, src_seq_len, kv_heads, head_dim]
  
  float max_val = -1e9f;
  float sum_exp = 0.0f;
  float output_acc = 0.0f;
  
  // First pass: find max for numerical stability
  for (int s = 0; s < src_seq_len; s++) {
    // Causal mask: only attend to positions <= t
    if (is_causal && s > t) continue;
    
    // Compute Q[b,t,h,:] dot K[b,s,kv_h,:]
    float dot = 0.0f;
    for (int dd = 0; dd < head_dim; dd++) {
      int q_idx = ((b * tgt_seq_len + t) * query_heads + h) * head_dim + dd;
      int k_idx = ((b * src_seq_len + s) * kv_heads + kv_h) * head_dim + dd;
      dot += TtoFloat(Q[q_idx]) * TtoFloat(K[k_idx]);
    }
    dot *= scale;
    max_val = fmaxf(max_val, dot);
  }
  
  // Second pass: compute softmax and weighted sum
  for (int s = 0; s < src_seq_len; s++) {
    if (is_causal && s > t) continue;
    
    // Recompute dot product
    float dot = 0.0f;
    for (int dd = 0; dd < head_dim; dd++) {
      int q_idx = ((b * tgt_seq_len + t) * query_heads + h) * head_dim + dd;
      int k_idx = ((b * src_seq_len + s) * kv_heads + kv_h) * head_dim + dd;
      dot += TtoFloat(Q[q_idx]) * TtoFloat(K[k_idx]);
    }
    dot *= scale;
    
    float exp_val = expf(dot - max_val);
    sum_exp += exp_val;
    
    // Accumulate weighted V
    int v_idx = ((b * src_seq_len + s) * kv_heads + kv_h) * head_dim + d;
    output_acc += exp_val * TtoFloat(V[v_idx]);
  }
  
  // Normalize and write output
  int o_idx = ((b * tgt_seq_len + t) * query_heads + h) * head_dim + d;
  if (sum_exp > 0.0f) {
    O[o_idx] = floatToT<T>(output_acc / sum_exp);
  } else {
    O[o_idx] = floatToT<T>(0.0f);
  }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // Calculate sizes
  size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
  size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
  size_t o_size = q_size;
  
  // Ensure output vector is properly sized
  h_o.resize(o_size);
  
  // Allocate device memory
  T *d_q, *d_k, *d_v, *d_o;
  cudaMalloc(&d_q, q_size * sizeof(T));
  cudaMalloc(&d_k, kv_size * sizeof(T));
  cudaMalloc(&d_v, kv_size * sizeof(T));
  cudaMalloc(&d_o, o_size * sizeof(T));
  
  // Copy inputs to device
  cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
  
  // Scale factor: 1/sqrt(head_dim)
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  
  // Launch kernel
  // Grid: one block per (batch, target_pos, head) combination
  int num_outputs = batch_size * target_seq_len * query_heads;
  dim3 grid(num_outputs, (head_dim + 31) / 32);
  dim3 block(32);  // 32 threads per block for head_dim dimension
  
  flashAttentionKernel<T><<<grid, block>>>(
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim,
      is_causal, scale);
  
  // Copy result back
  cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);
  
  // Cleanup
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);

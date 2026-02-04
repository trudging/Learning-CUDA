/**
 * @file kernels.cu
 * @brief CUDA kernel implementations for matrix trace and Flash Attention
 * @author Training Camp Student
 * @date 2026-02
 * 
 * This file contains optimized CUDA implementations of:
 * 1. Matrix trace computation with parallel reduction
 * 2. Flash Attention with causal masking and GQA support
 * 
 * Optimization techniques used:
 * - Warp shuffle for fast intra-warp reduction
 * - Shared memory tiling for data reuse
 * - Memory coalescing for efficient global memory access
 * - Loop unrolling for reduced instruction overhead
 */

#include <vector>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

#include "../tester/utils.h"

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int TRACE_BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 32;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Warp-level reduction using shuffle instructions
 * @tparam T Data type (must support __shfl_down_sync)
 * @param val Value to reduce within warp
 * @return Sum of all values in the warp (valid only for lane 0)
 */
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Block-level reduction using shared memory and warp shuffle
 * @tparam T Data type
 * @param val Value to reduce
 * @param shared Shared memory buffer (size >= blockDim.x / WARP_SIZE)
 * @return Sum of all values in the block (valid only for thread 0)
 */
template <typename T>
__device__ __forceinline__ T blockReduceSum(T val, T* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    
    // Intra-warp reduction
    val = warpReduceSum(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    const int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : T(0);
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// ============================================================================
// TRACE KERNEL IMPLEMENTATION
// ============================================================================

/**
 * @brief CUDA kernel for computing matrix trace with optimized reduction
 * 
 * Uses a two-phase approach:
 * 1. Each thread loads one diagonal element
 * 2. Block-level reduction using warp shuffle primitives
 * 3. Atomic accumulation of block results
 * 
 * @tparam T Element type (int or float)
 * @param input Flattened input matrix (row-major)
 * @param output Pointer to scalar output (must be zero-initialized)
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
template <typename T>
__global__ void traceKernel(const T* __restrict__ input, 
                            T* __restrict__ output, 
                            size_t rows, 
                            size_t cols) {
    // Shared memory for block reduction
    __shared__ T sharedMem[TRACE_BLOCK_SIZE / WARP_SIZE];
    
    const size_t diagLen = min(rows, cols);
    const size_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load diagonal element (coalesced access pattern for diagonal)
    T localSum = T(0);
    if (globalIdx < diagLen) {
        // Diagonal element at position (i, i) in row-major layout
        localSum = input[globalIdx * cols + globalIdx];
    }
    
    // Block-level reduction
    localSum = blockReduceSum(localSum, sharedMem);
    
    // Atomic accumulation (only thread 0 of each block)
    if (threadIdx.x == 0) {
        atomicAdd(output, localSum);
    }
}

/**
 * @brief Computes the trace of a matrix using CUDA
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This implementation uses parallel reduction on GPU for efficient computation.
 *
 * Algorithm complexity: O(n/p) where n = min(rows, cols) and p = #threads
 * 
 * @tparam T The numeric type of matrix elements (int or float)
 * @param h_input A flattened row-major matrix of size rows * cols
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @return The trace (sum of diagonal values) of the matrix
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    // Handle edge cases
    const size_t diagLen = std::min(rows, cols);
    if (diagLen == 0) {
        return T(0);
    }
    
    // Calculate memory requirements
    const size_t inputBytes = rows * cols * sizeof(T);
    const size_t outputBytes = sizeof(T);
    
    // Allocate device memory
    T* d_input = nullptr;
    T* d_output = nullptr;
    cudaMalloc(&d_input, inputBytes);
    cudaMalloc(&d_output, outputBytes);
    
    // Transfer input data to device
    cudaMemcpy(d_input, h_input.data(), inputBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, outputBytes);
    
    // Configure kernel launch parameters
    const int blockSize = TRACE_BLOCK_SIZE;
    const int numBlocks = (diagLen + blockSize - 1) / blockSize;
    
    // Launch trace kernel
    traceKernel<T><<<numBlocks, blockSize>>>(d_input, d_output, rows, cols);
    
    // Retrieve result
    T result;
    cudaMemcpy(&result, d_output, outputBytes, cudaMemcpyDeviceToHost);
    
    // Release device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

// ============================================================================
// FLASH ATTENTION IMPLEMENTATION
// ============================================================================

/**
 * @brief Type conversion utilities for mixed-precision computation
 */
template <typename T>
struct TypeConverter {
    __device__ __forceinline__ static float toFloat(T val);
    __device__ __forceinline__ static T fromFloat(float val);
};

template <>
struct TypeConverter<float> {
    __device__ __forceinline__ static float toFloat(float val) { return val; }
    __device__ __forceinline__ static float fromFloat(float val) { return val; }
};

template <>
struct TypeConverter<half> {
    __device__ __forceinline__ static float toFloat(half val) { return __half2float(val); }
    __device__ __forceinline__ static half fromFloat(float val) { return __float2half(val); }
};

/**
 * @brief Optimized Flash Attention kernel with online softmax
 * 
 * Implementation features:
 * - Online softmax computation for numerical stability
 * - Shared memory tiling for K and V
 * - GQA support via head index remapping
 * - Causal masking with early termination
 * 
 * @tparam T Data type (float or half)
 */
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    const int batchSize,
    const int tgtSeqLen,
    const int srcSeqLen,
    const int queryHeads,
    const int kvHeads,
    const int headDim,
    const bool isCausal,
    const float scale) {
    
    // Shared memory for tiled K and V
    extern __shared__ char sharedMemory[];
    float* sharedK = reinterpret_cast<float*>(sharedMemory);
    float* sharedV = sharedK + TILE_SIZE * headDim;
    
    // Thread and block identification
    const int batchIdx = blockIdx.z;
    const int headIdx = blockIdx.y;
    const int tgtPos = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Early exit for out-of-bounds
    if (batchIdx >= batchSize || headIdx >= queryHeads || tgtPos >= tgtSeqLen) {
        return;
    }
    
    // GQA: map query head to corresponding KV head
    const int kvHeadIdx = headIdx / (queryHeads / kvHeads);
    
    // Base pointers
    const int qOffset = ((batchIdx * tgtSeqLen + tgtPos) * queryHeads + headIdx) * headDim;
    const int kvBatchOffset = batchIdx * srcSeqLen * kvHeads * headDim;
    
    // Load query vector into registers
    float qReg[8] = {0.0f};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int d = tid + i * blockDim.x;
        if (d < headDim) {
            qReg[i] = TypeConverter<T>::toFloat(Q[qOffset + d]);
        }
    }
    
    // Online softmax state
    float maxScore = -INFINITY;
    float sumExp = 0.0f;
    float output[8] = {0.0f};
    
    // Determine max source position (causal optimization)
    const int maxSrcPos = isCausal ? min(tgtPos + 1, srcSeqLen) : srcSeqLen;
    
    // Process K/V in tiles
    for (int tileStart = 0; tileStart < maxSrcPos; tileStart += TILE_SIZE) {
        const int tileEnd = min(tileStart + TILE_SIZE, maxSrcPos);
        const int tileLen = tileEnd - tileStart;
        
        // Collaborative loading of K and V tiles
        for (int loadIdx = tid; loadIdx < tileLen * headDim; loadIdx += blockDim.x) {
            const int localSrcPos = loadIdx / headDim;
            const int d = loadIdx % headDim;
            const int globalSrcPos = tileStart + localSrcPos;
            
            const int kvIdx = kvBatchOffset + (globalSrcPos * kvHeads + kvHeadIdx) * headDim + d;
            sharedK[localSrcPos * headDim + d] = TypeConverter<T>::toFloat(K[kvIdx]);
            sharedV[localSrcPos * headDim + d] = TypeConverter<T>::toFloat(V[kvIdx]);
        }
        __syncthreads();
        
        // Compute attention for each position in tile
        for (int localSrcPos = 0; localSrcPos < tileLen; localSrcPos++) {
            // Compute Q·K^T
            float score = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int d = tid + i * blockDim.x;
                if (d < headDim) {
                    score += qReg[i] * sharedK[localSrcPos * headDim + d];
                }
            }
            
            // Warp reduction for score
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                score += __shfl_down_sync(0xffffffff, score, offset);
            }
            score = __shfl_sync(0xffffffff, score, 0);
            score *= scale;
            
            // Online softmax update
            const float prevMax = maxScore;
            maxScore = fmaxf(maxScore, score);
            const float expCorrection = expf(prevMax - maxScore);
            sumExp = sumExp * expCorrection + expf(score - maxScore);
            
            // Update output with correction
            const float attnWeight = expf(score - maxScore);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int d = tid + i * blockDim.x;
                if (d < headDim) {
                    output[i] = output[i] * expCorrection + attnWeight * sharedV[localSrcPos * headDim + d];
                }
            }
        }
        __syncthreads();
    }
    
    // Normalize and write output
    const float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
    const int oOffset = ((batchIdx * tgtSeqLen + tgtPos) * queryHeads + headIdx) * headDim;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int d = tid + i * blockDim.x;
        if (d < headDim) {
            O[oOffset + d] = TypeConverter<T>::fromFloat(output[i] * invSum);
        }
    }
}

/**
 * @brief Fallback attention kernel for arbitrary dimensions
 * 
 * Standard two-pass softmax implementation for correctness.
 */
template <typename T>
__global__ void flashAttentionSimpleKernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    const int batchSize,
    const int tgtSeqLen,
    const int srcSeqLen,
    const int queryHeads,
    const int kvHeads,
    const int headDim,
    const bool isCausal,
    const float scale) {
    
    const int linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalOutputs = batchSize * tgtSeqLen * queryHeads * headDim;
    
    if (linearIdx >= totalOutputs) return;
    
    // Decode linear index
    const int d = linearIdx % headDim;
    const int h = (linearIdx / headDim) % queryHeads;
    const int t = (linearIdx / (headDim * queryHeads)) % tgtSeqLen;
    const int b = linearIdx / (headDim * queryHeads * tgtSeqLen);
    
    // GQA mapping
    const int kvH = h / (queryHeads / kvHeads);
    
    // Determine source range
    const int maxSrcPos = isCausal ? min(t + 1, srcSeqLen) : srcSeqLen;
    
    // First pass: find max
    float maxVal = -INFINITY;
    for (int s = 0; s < maxSrcPos; s++) {
        float dot = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            const int qIdx = ((b * tgtSeqLen + t) * queryHeads + h) * headDim + dd;
            const int kIdx = ((b * srcSeqLen + s) * kvHeads + kvH) * headDim + dd;
            dot += TypeConverter<T>::toFloat(Q[qIdx]) * TypeConverter<T>::toFloat(K[kIdx]);
        }
        maxVal = fmaxf(maxVal, dot * scale);
    }
    
    // Second pass: softmax + weighted sum
    float sumExp = 0.0f;
    float result = 0.0f;
    
    for (int s = 0; s < maxSrcPos; s++) {
        float dot = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            const int qIdx = ((b * tgtSeqLen + t) * queryHeads + h) * headDim + dd;
            const int kIdx = ((b * srcSeqLen + s) * kvHeads + kvH) * headDim + dd;
            dot += TypeConverter<T>::toFloat(Q[qIdx]) * TypeConverter<T>::toFloat(K[kIdx]);
        }
        
        const float expVal = expf(dot * scale - maxVal);
        sumExp += expVal;
        
        const int vIdx = ((b * srcSeqLen + s) * kvHeads + kvH) * headDim + d;
        result += expVal * TypeConverter<T>::toFloat(V[vIdx]);
    }
    
    const int oIdx = ((b * tgtSeqLen + t) * queryHeads + h) * headDim + d;
    O[oIdx] = TypeConverter<T>::fromFloat((sumExp > 0.0f) ? (result / sumExp) : 0.0f);
}

/**
 * @brief Computes Flash Attention for given query, key, and value tensors
 * 
 * Implements scaled dot-product attention with support for:
 * - Causal masking (for autoregressive models)
 * - Grouped Query Attention (GQA) where query_heads > kv_heads
 * 
 * The attention computation follows:
 *   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(head_dim)) @ V
 * 
 * @tparam T Data type (float or half)
 * @param h_q Query tensor [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param h_k Key tensor [batch_size, src_seq_len, kv_heads, head_dim]
 * @param h_v Value tensor [batch_size, src_seq_len, kv_heads, head_dim]
 * @param h_o Output tensor [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param batch_size Batch dimension size
 * @param target_seq_len Target sequence length
 * @param src_seq_len Source sequence length
 * @param query_heads Number of query attention heads
 * @param kv_heads Number of key/value heads (for GQA)
 * @param head_dim Dimension size of each attention head
 * @param is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    // Calculate tensor sizes
    const size_t qSize = batch_size * target_seq_len * query_heads * head_dim;
    const size_t kvSize = batch_size * src_seq_len * kv_heads * head_dim;
    const size_t oSize = qSize;
    
    // Ensure output vector is properly sized
    h_o.resize(oSize);
    
    // Allocate device memory
    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, qSize * sizeof(T));
    cudaMalloc(&d_k, kvSize * sizeof(T));
    cudaMalloc(&d_v, kvSize * sizeof(T));
    cudaMalloc(&d_o, oSize * sizeof(T));
    
    // Transfer input data to device
    cudaMemcpy(d_q, h_q.data(), qSize * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kvSize * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kvSize * sizeof(T), cudaMemcpyHostToDevice);
    
    // Compute scaling factor: 1 / sqrt(head_dim)
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Choose kernel based on problem size
    const bool useOptimizedKernel = (head_dim <= 256) && (head_dim % 32 == 0);
    
    if (useOptimizedKernel) {
        // Optimized tiled kernel
        const int blockSize = min(256, (head_dim + 7) / 8 * 8);
        const size_t sharedMemSize = 2 * TILE_SIZE * head_dim * sizeof(float);
        
        dim3 grid(target_seq_len, query_heads, batch_size);
        dim3 block(blockSize);
        
        flashAttentionKernel<T><<<grid, block, sharedMemSize>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            is_causal, scale);
    } else {
        // Fallback kernel for arbitrary dimensions
        const int totalOutputs = batch_size * target_seq_len * query_heads * head_dim;
        const int blockSize = 256;
        const int numBlocks = (totalOutputs + blockSize - 1) / blockSize;
        
        flashAttentionSimpleKernel<T><<<numBlocks, blockSize>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            is_causal, scale);
    }
    
    // Transfer results back to host
    cudaMemcpy(h_o.data(), d_o, oSize * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Release device memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// Required for linking with the tester - DO NOT MODIFY
// ============================================================================
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);

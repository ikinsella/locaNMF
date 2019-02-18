#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

//TODO detect hardware details using cuda runtime
const size_t WARP_SIZE = 32;
const size_t MAX_WARPS_PER_SM = 64; 
const size_t MAX_SMEM_PER_SM = 96 * 1024;

template<typename scalar_t>
__device__ __forceinline__
scalar_t safe_inverse_var(scalar_t var) {
    return (var > 0) ? (1 / var) : 0;
}

template<typename scalar_t>
__global__ 
void updateKernel(int const N, int const P,
                  scalar_t      *const __restrict__ g_Beta,  int const ldBeta,
                  scalar_t const*const __restrict__ g_Delta, int const ldDelta,
                  scalar_t const*const __restrict__ g_Sigma, int const ldSigma,
                  bool const nonnegative) 
{
    // Shared Variable Declaration
    extern __shared__ scalar_t smem[];             // N*W + W + N + 1
    scalar_t* s_B = &smem[0];                      // NxW Row Major 
    scalar_t* s_Dn = &smem[N*blockDim.x];          // W Vec
    scalar_t* s_Sn = &smem[(N+1)*blockDim.x];      // N Vec
    scalar_t* s_scale = &smem[(N+1)*blockDim.x+N]; // Scalar 
    
    // Offsets For Accessing Shared & Local Mem
    const int thread_id = threadIdx.x + (threadIdx.y * blockDim.x); 
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Read Slice Of Beta
    __syncthreads();
    if(p < P){
        for (int n = threadIdx.y; n < N; n += blockDim.y)
            s_B[n * blockDim.x + threadIdx.x] = g_Beta[n * ldBeta + p];
    }

    // Update Each Component To Fit Residual While Holding Others Fixed 
    for (int n=0; n < N; n++)
    {
        // Read Active-Component Slices Of Delta, Sigma
        __syncthreads();
        if (threadIdx.y == 0 && p < P) s_Dn[threadIdx.x] = g_Delta[n * ldBeta + p];
        __syncthreads();
        if (thread_id < N) s_Sn[thread_id] = g_Sigma[n * ldSigma + thread_id]; 
        if (thread_id == n) s_scale[0] = safe_inverse_var(s_Sn[thread_id]); 

        // Correct Residual For Inactive Components
        __syncthreads();
        scalar_t accum = 0.0;
        if (p < P){
            for (int ndx = threadIdx.y; ndx < N ; ndx += blockDim.y)
                accum -= s_Sn[ndx] * s_B[ndx * blockDim.x + threadIdx.x];
        }
        __syncthreads();
        // TODO: Restrict Atomic To Block
        if (p < P) atomicAdd(&s_Dn[threadIdx.x], accum);

        // Perform (Independent) Scalar Updates
        __syncthreads();
        if (threadIdx.y == 0 && p < P){
            const scalar_t beta = s_B[n * blockDim.x + threadIdx.x] + s_Dn[threadIdx.x] * s_scale[0];
            s_B[n*blockDim.x + threadIdx.x] = (nonnegative && beta < 0.0) ? 0.0 : beta; 
        }
    }

    // Write Slice Of Beta
    __syncthreads();
    if (p < P){
        for (int n = threadIdx.y; n < N; n += blockDim.y)
            g_Beta[n * ldBeta + p] = s_B[n * blockDim.x + threadIdx.x];
    }
}

void gpuUpdate(int const N, int const P,
               float      *const __restrict__ Beta,  int const ldBeta,
               float const*const __restrict__ Delta, int const ldDelta,
               float const*const __restrict__ Sigma, int const ldSigma,
               bool const nonnegative)
{
// Compute Grid & Block Layouts
const size_t smem = sizeof(float) * (WARP_SIZE + 1) * (N + 1);
const size_t warps_per_block = (smem * MAX_WARPS_PER_SM < MAX_SMEM_PER_SM) ? 
(1) : (MAX_WARPS_PER_SM / (MAX_SMEM_PER_SM / smem));
const dim3 block(WARP_SIZE, warps_per_block);
const dim3 grid((P + block.x - 1) / block.x);

// TODO: explicitly enforce Block Size > N

// Dispatch Kernel
updateKernel<float><<<grid, block, smem>>>(N, P,
                                           Beta, ldBeta,
                                           Delta, ldDelta,
                                           Sigma, ldSigma,
                                           nonnegative);
}

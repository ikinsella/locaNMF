#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

//TODO detect hardware details using cuda runtime
const size_t WARP_SIZE = 32;
const size_t MAX_WARPS_PER_SM = 64; 
const size_t MAX_SMEM_PER_SM = 96 * 1024;

template <typename scalar_t>
__device__ __forceinline__
scalar_t* shared_memory_proxy()
{
    extern __shared__ double2 memory[];
    return reinterpret_cast<scalar_t*>(memory);
}

template<typename scalar_t>
__global__ 
void updateKernel(int const N, int const P,
                  scalar_t      *const __restrict__ g_Beta,  int const ldBeta,
                  scalar_t const*const __restrict__ g_Delta, int const ldDelta,
                  scalar_t const*const __restrict__ g_Sigma, int const ldSigma,
                  bool const nonnegative) 
{
    // Offsets For Accessing Shared & Local Mem
    const int thread_id = threadIdx.x + (threadIdx.y * blockDim.x); 
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
 
    // Shared Variable Declaration 
    scalar_t* smem = shared_memory_proxy<scalar_t>();  // (T+2)*N + T Floats Needed
    scalar_t* s_B = &smem[0];                          // NxW Row Major 
    scalar_t* s_Dn = &smem[N*blockDim.x];              // W Vec
    scalar_t* s_Sn = &smem[(N+1)*blockDim.x];          // N Vec
    scalar_t* s_scale = &smem[(N+1)*blockDim.x+N];     // N Vec
    
    // Copy All Of Scale To Shared Memory Early 
    __syncthreads();
    if (thread_id < N){
        const scalar_t var = g_Sigma[thread_id * ldSigma + thread_id];
        s_scale[thread_id] = (var > 0) ? (1 / var) : 0; 
    }
    // Copy Block's Slice Of Beta To Shared Memory
    __syncthreads();
    if(p < P){
        for (int n = threadIdx.y; n < N; n += blockDim.y)
            s_B[n * blockDim.x + threadIdx.x] = g_Beta[n * ldBeta + p];
    }
    // Update all N components sequentially, sync across slice of P
    for (int n=0; n < N; n++)
    {
        // Split Global -> Shared Memory Reads Across Thread Pool
        __syncthreads();
        if (threadIdx.y == 0 && p < P){
            s_Dn[threadIdx.x] = g_Delta[n * ldBeta + p];
        }
        __syncthreads();
        if (thread_id < N){
            s_Sn[thread_id] = g_Sigma[n * ldSigma + thread_id]; 
        } 
        // Matvec(Beta, Sigma[ndx])
        __syncthreads();
        scalar_t accum = 0.0;
        if (p < P){
            for (int ndx = threadIdx.y; ndx < N ; ndx += blockDim.y)
                accum -= s_Sn[ndx] * s_B[ndx * blockDim.x + threadIdx.x];
        }
        __syncthreads();
        if (p < P){
            atomicAdd(&s_Dn[threadIdx.x], accum);
        }
        // Perform Scalar Update In Smem
        __syncthreads();
        if (threadIdx.y == 0 && p < P){
            const scalar_t beta = s_B[n * blockDim.x + threadIdx.x] + s_Dn[threadIdx.x] * s_scale[n];
            s_B[n*blockDim.x + threadIdx.x] = (nonnegative && beta < 0.0) ? 0.0 : beta; 
        }
    }
    // Copy Slice Of Beta From Shared Memory Back To Global
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
const size_t smem = sizeof(float) * ((WARP_SIZE + 2) * N + WARP_SIZE);
const size_t warps_per_block = (smem * MAX_WARPS_PER_SM < MAX_SMEM_PER_SM) ? 
(1) : (MAX_WARPS_PER_SM / (MAX_SMEM_PER_SM / smem));
const dim3 block(WARP_SIZE, warps_per_block);
const dim3 grid((P + block.x - 1) / block.x);

// Dispatch Kernel
updateKernel<float><<<grid, block, smem>>>(N, P,
                                           Beta, ldBeta,
                                           Delta, ldDelta,
                                           Sigma, ldSigma,
                                           nonnegative);
}

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mkl.h>
#include <math.h>
#include <omp.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_MATRIX(x) CHECK_CONTIGUOUS(x); AT_ASSERTM(x.dim() == 2, #x "must be dim2")
#define CHECK_CUDA_MATRIX(x) CHECK_CUDA(x); CHECK_MATRIX(x)

/* Cuda Kernel Dispatcher Defined In .cu File */
void gpuUpdate(int const N, int const P,
               float*       __restrict__ Beta,  int const ldBeta,
               float* const __restrict__ Delta, int const ldDelta,
               float* const __restrict__ Sigma, int const ldSigma,
               bool const nonnegative);

/* SIMD Parallelizable CPU Update Function */
#pragma omp declare simd notinbranch
float rect(float val) { return val > 0.0 ? val : 0; }

/* Sequential HALS Update Of One Coefficient Matrix Given Precomputed Quantities
 *    Sigma: NxN Precomputed Covariance Of Design Matrix X
 *    Beta: NxP Row Major or PxN Col Major Coefficient Matrix To Be Updated 
 *    Delta: NxP Row Major or PxN Col Major Precomputed Updates (YtX, XtY)
 */
void seqCpuUpdate(int const N, int const P,
                  float*       __restrict__ Beta,  int const ldBeta,
                  float* const __restrict__ Delta, int const ldDelta,
                  float* const __restrict__ Sigma, int const ldSigma,
                  bool const nonnegative)
{
    for (int ndx = 0; ndx < N; ndx++)
    {
        float* Dn = Delta + ldDelta * ndx;
        const float var = Sigma[ndx * ldSigma + ndx];
        cblas_sgemv(CblasColMajor, CblasNoTrans,
                    P, N,
                    -1.0 / var,
                    Beta, ldBeta,
                    Sigma + ndx * ldSigma, 1,
                    1.0 / var,
                    Dn, 1);
        float* Bn = Beta + ldBeta * ndx;
        if (nonnegative){
#pragma omp simd
            for (int pdx = 0; pdx < P; pdx++)
                Bn[pdx] = rect(Bn[pdx] + Dn[pdx]);
        } else {
#pragma omp simd
            for (int pdx = 0; pdx < P; pdx++)
                Bn[pdx] = Bn[pdx] + Dn[pdx];
        }
    }
}

/* Parallelized HALS Update Of One Coefficient Matrix Given Precomputed Quantities
 *    Sigma: NxN Precomputed Covariance Of Design Matrix X
 *    Beta: NxP Row Major or PxN Col Major Coefficient Matrix To Be Updated
 *    Delta: NxP Row Major or PxN Col Major Precomputed Updates (YtX, XtY)
 */
void cpuUpdate(int const N, int const P,
               float*       __restrict__ Beta,  int const ldBeta,
               float* const __restrict__ Delta, int const ldDelta,
               float* const __restrict__ Sigma, int const ldSigma,
               bool const nonnegative)
{
#pragma omp parallel default(shared)
    {
        const int Pstart = (omp_get_thread_num() * P) / omp_get_num_threads();
        const int Pend = ((omp_get_thread_num() + 1) * P) / omp_get_num_threads();
        seqCpuUpdate(N, Pend - Pstart,
                     Beta + Pstart, ldBeta,
                     Delta + Pstart, ldDelta,
                     Sigma, ldSigma,
                     nonnegative);
    }
}

void update(const at::Tensor& Beta,
            const at::Tensor& Delta,
            const at::Tensor& Sigma,
            const bool nonnegative=true,
            const int64_t batch=32)
{
    // Validate Input Tensors
    const bool cuda = Beta.is_cuda();
    CHECK_MATRIX(Beta);
    if (cuda){
        CHECK_CUDA_MATRIX(Delta);
        CHECK_CUDA_MATRIX(Sigma);
    } else {
        CHECK_MATRIX(Delta);
        CHECK_MATRIX(Sigma);
    }

    // Identify Batch Size
    const int ncomp = Beta.size(0);
    const int nfeat = Beta.size(1);
    const int ldBeta = Beta.stride(0);
    const int ldDelta = Delta.stride(0);
    const int ldSigma = Sigma.stride(0);
    for (int first=0; first < ncomp; first += batch)
    {
        const int last = (ncomp < first + batch) ? ncomp : first + batch;
        auto B_pi = Beta.narrow(0, first, last - first);
        auto D_pi = Delta.narrow(0, first, last - first);
        auto S_pi = Sigma.narrow(0, first, last - first);
        // Delta[first:last] -= Sigma[first:last,:first] @ Beta[:first]
        if (first > 0)
            at::addmm_out(D_pi, D_pi,
                          S_pi.narrow(1, 0, first),
                          Beta.narrow(0, 0, first),
                          1.0,
                          -1.0);
        // Delta[first:last] -= Sigma[first:last,last:] @ Beta[last:]
        if (last < ncomp)
            at::addmm_out(D_pi, D_pi,
                          S_pi.narrow(1, last, ncomp - last),
                          Beta.narrow(0, last, ncomp - last),
                          1.0,
                          -1.0);
        // Perform Hals Within Partition
        if (cuda){
            gpuUpdate(last - first, nfeat,
                      Beta.data<float>() + first * ldBeta, ldBeta,
                      Delta.data<float>() + first * ldDelta, ldDelta,
                      Sigma.data<float>() + first * ldSigma + first, ldSigma,
                      nonnegative);
        } else {
            cpuUpdate(last - first, nfeat,
                      Beta.data<float>() + first * ldBeta, ldBeta,
                      Delta.data<float>() + first * ldDelta, ldDelta,
                      Sigma.data<float>() + first * ldSigma + first, ldSigma,
                      nonnegative);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("update",
            &update,
            "Fast HALS Update On CPU/GPU.");
}
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void launchUpdateKernel(const int N,
                        const int P,
                        float* __restrict__ Beta,
                        const float* __restrict__ Delta,
                        const int ldBeta,
                        const float* __restrict__ Sigma,
                        const float* __restrict__ scale,
                        const int ldSigma,
                        const bool nonnegative);

void update(const at::Tensor& Beta,
            const at::Tensor& Delta,
            const at::Tensor& Sigma,
            const at::Tensor& scale,
            const bool nonnegative=true,
            const int64_t batch=32)
{
    // Validate Input Tensors
    CHECK_INPUT(Beta);
    CHECK_INPUT(Delta);
    CHECK_INPUT(Sigma);
    CHECK_INPUT(scale);

    // Identify Batch Size
    const int64_t ncomp = Beta.size(0);
    const int64_t nfeat = Beta.size(1);
    const int64_t ldBeta = Beta.stride(0);
    const int64_t ldSigma = Sigma.stride(0);
    for (int64_t first=0; first < ncomp; first += batch)
    {
        const int64_t last = (ncomp < first + batch) ? ncomp : first + batch;
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
        launchUpdateKernel(last - first,
                           nfeat,
                           Beta.data<float>() + first * ldBeta,
                           Delta.data<float>() + first * ldBeta,
                           ldBeta,
                           Sigma.data<float>() + first * ldSigma + first,
                           scale.data<float>() + first,
                           ldSigma,
                           nonnegative);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("update",  &update, "Fast HALS Update Kernel. Uses Precomputed Quantities To Perform Updates Inplace.");
}

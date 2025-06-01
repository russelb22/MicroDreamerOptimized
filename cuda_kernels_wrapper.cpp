#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel declaration
extern "C" void launch_gaussian_3d_coeff_gpu(
    const float* xyzs,
    const float* covs,
    float* out,
    int N
);

// Wrapper
void gaussian_3d_launcher(torch::Tensor xyzs, torch::Tensor covs, torch::Tensor out) {
    int N = xyzs.size(0);

    launch_gaussian_3d_coeff_gpu(
        xyzs.data_ptr<float>(),
        covs.data_ptr<float>(),
        out.data_ptr<float>(),
        N
    );
}

// Pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_3d_launcher", &gaussian_3d_launcher, "Launch gaussian_3d_launcher kernel");
}
#include <torch/extension.h>
#include <cuda_runtime.h>

// Declarations of CUDA launchers

extern "C" void launch_gaussian_3d_coeff_gpu(
    const float* xyzs,
    const float* covs,
    float* out,
    int          N
);

extern "C" void launch_extract_fields(
    const float* d_means,      // [N0 x 3]
    const float* d_inv_cov6,   // [N0 x 6]
    const float* d_opacity,    // [N0]
    int          N0,
    int          resolution,
    int          num_blocks,
    float        relax_ratio,
    float* d_occ         // [resolution^3]
);

// Wrapper for gaussian_3d_coeff

void gaussian_3d_launcher(
    torch::Tensor xyzs,   // [N x 3], float32 CUDA
    torch::Tensor covs,   // [N x 6], float32 CUDA
    torch::Tensor out     // [N],    float32 CUDA (pre-allocated)
) {
    int N = xyzs.size(0);

    TORCH_CHECK(xyzs.is_cuda(), "xyzs must be a CUDA tensor");
    TORCH_CHECK(covs.is_cuda(), "covs must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(xyzs.dtype() == torch::kFloat32, "xyzs must be float32");
    TORCH_CHECK(covs.dtype() == torch::kFloat32, "covs must be float32");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(xyzs.is_contiguous(), "xyzs must be contiguous");
    TORCH_CHECK(covs.is_contiguous(), "covs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    launch_gaussian_3d_coeff_gpu(
        xyzs.data_ptr<float>(),
        covs.data_ptr<float>(),
        out.data_ptr<float>(),
        N
    );
}

// Wrapper for extract_fields

torch::Tensor extract_fields_launcher(
    torch::Tensor means,       // [N0 x 3], float32 CUDA
    torch::Tensor inv_cov6,    // [N0 x 6], float32 CUDA
    torch::Tensor opacities,   // [N0],     float32 CUDA
    int           resolution,  // e.g. 128
    int           num_blocks,  // e.g. 16 (so split_size = resolution/num_blocks)
    float         relax_ratio  // e.g. 1.5
) {
    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK(inv_cov6.is_cuda(), "inv_cov6 must be a CUDA tensor");
    TORCH_CHECK(opacities.is_cuda(), "opacities must be a CUDA tensor");
    TORCH_CHECK(means.dtype() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(inv_cov6.dtype() == torch::kFloat32, "inv_cov6 must be float32");
    TORCH_CHECK(opacities.dtype() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
    TORCH_CHECK(inv_cov6.is_contiguous(), "inv_cov6 must be contiguous");
    TORCH_CHECK(opacities.is_contiguous(), "opacities must be contiguous");

    int64_t N0 = means.size(0);
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
        "means must have shape [N0,3]");
    TORCH_CHECK(inv_cov6.dim() == 2 && inv_cov6.size(0) == N0 && inv_cov6.size(1) == 6,
        "inv_cov6 must have shape [N0,6]");
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N0,
        "opacities must have shape [N0]");

    auto opts = means.options();  // same device and dtype as means
    int64_t total_voxels = int64_t(resolution) * resolution * resolution;
    torch::Tensor occ_flat = torch::zeros({ total_voxels }, opts);

    launch_extract_fields(
        means.data_ptr<float>(),
        inv_cov6.data_ptr<float>(),
        opacities.data_ptr<float>(),
        static_cast<int>(N0),
        resolution,
        num_blocks,
        relax_ratio,
        occ_flat.data_ptr<float>()
    );

    return occ_flat.view({ resolution, resolution, resolution });
}

// PyBind11 module definition

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gaussian_3d_launcher",
        &gaussian_3d_launcher,
        "Launch gaussian_3d_coeff CUDA kernel"
    );
    m.def(
        "extract_fields_launcher",
        &extract_fields_launcher,
        "Launch fused extract_fields CUDA kernel"
    );
}
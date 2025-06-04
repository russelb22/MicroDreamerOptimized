# cuda_wrapper.py

import torch
import cuda_kernels  # the compiled extension

def gaussian_3d_coeff_gpu(xyzs, covs):
    """
    xyzs: Tensor[N,3]
    covs: Tensor[N,6]
    returns: Tensor[N] of exp(-0.5 * (x^T * Sigma^-1 * x))
    """
    assert xyzs.shape[0] == covs.shape[0], "Mismatched shapes"
    assert xyzs.shape[1] == 3 and covs.shape[1] == 6, "Invalid input shapes"

    xyzs = xyzs.contiguous().to("cuda", dtype=torch.float32)
    covs = covs.contiguous().to("cuda", dtype=torch.float32)
    out  = torch.empty(xyzs.shape[0], device="cuda", dtype=torch.float32)

    cuda_kernels.gaussian_3d_launcher(xyzs, covs, out)
    return out


def extract_fields_gpu(means, inv_cov6, opacities, resolution, num_blocks, relax_ratio=1.5):
    """
    means:     Tensor[N0,3]    (Gaussian centers, in normalized [-1,1] space)
    inv_cov6:  Tensor[N0,6]    (packed inverse covariance per Gaussian)
    opacities: Tensor[N0]      (per-Gaussian opacity)
    resolution: int            (e.g. 128)
    num_blocks: int            (e.g. 16, so split_size = resolution//num_blocks)
    relax_ratio: float         (e.g. 1.5)

    returns: occ (Tensor[resolution, resolution, resolution]) on CUDA
    """
    N0 = means.size(0)
    # shape checks
    assert means.ndim == 2 and means.size(1) == 3,       "means must be [N0,3]"
    assert inv_cov6.ndim == 2 and inv_cov6.size(1) == 6, "inv_cov6 must be [N0,6]"
    assert opacities.ndim == 1 and opacities.size(0) == N0, "opacities must be [N0]"

    means     = means.contiguous().to("cuda", dtype=torch.float32)
    inv_cov6  = inv_cov6.contiguous().to("cuda", dtype=torch.float32)
    opacities = opacities.contiguous().to("cuda", dtype=torch.float32)

    # Call the C++/CUDA launcher, which returns a flat [resolution^3] tensor reshaped inside
    occ3d = cuda_kernels.extract_fields_launcher(
        means,
        inv_cov6,
        opacities,
        resolution,
        num_blocks,
        float(relax_ratio)
    )
    return occ3d
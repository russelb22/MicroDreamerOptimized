import torch
import cuda_kernels  # This is the compiled extension

def gaussian_3d_coeff_gpu(xyzs, covs):
    assert xyzs.shape[0] == covs.shape[0], "Mismatched shapes"
    assert xyzs.shape[1] == 3 and covs.shape[1] == 6, "Invalid input shapes"

    xyzs = xyzs.contiguous().to("cuda", dtype=torch.float32)
    covs = covs.contiguous().to("cuda", dtype=torch.float32)
    out = torch.empty(xyzs.shape[0], device='cuda', dtype=torch.float32)

    cuda_kernels.gaussian_3d_launcher(xyzs, covs, out)

    return out

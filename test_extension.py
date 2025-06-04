# text_extension.py

import torch
import cuda_kernels  # This is your newly compiled extension

# -----------------------
# Test gaussian_3d_coeff
# -----------------------
N = 1024
xyzs = torch.randn(N, 3, device='cuda', dtype=torch.float32)
covs = torch.randn(N, 6, device='cuda', dtype=torch.float32)
out = torch.empty(N, device='cuda', dtype=torch.float32)

# Call the gaussian_3d_coeff kernel
cuda_kernels.gaussian_3d_launcher(xyzs.contiguous(), covs.contiguous(), out)

print("gaussian_3d_coeff output (first 10 values):")
print(out[:10])

# -----------------------------------
# Test extract_fields (fused 3D kernel)
# -----------------------------------
# For testing, choose a small resolution and number of Gaussians
resolution = 32
num_blocks = 4  # split_size = resolution // num_blocks = 8
relax_ratio = 1.5

# Create a small set of "active" Gaussians (N0)
N0 = 100
# Random Gaussian centers in [-1, 1]
means = (torch.rand(N0, 3, device='cuda', dtype=torch.float32) * 2.0) - 1.0

# For simplicity, create diagonal covariance matrices with small variances
# so that inverse covariance is just 1 / variance on the diagonal.
# Here we pack [inv_a, inv_b, inv_c, inv_d, inv_e, inv_f] for each Gaussian.
# We set inv_b = inv_c = inv_e = 0 for a diagonal covariance.
variances = torch.rand(N0, 3, device='cuda', dtype=torch.float32) * 0.1 + 0.05
# inv_a = 1/var_x, inv_d = 1/var_y, inv_f = 1/var_z
inv_a = 1.0 / variances[:, 0]
inv_b = torch.zeros(N0, device='cuda', dtype=torch.float32)
inv_c = torch.zeros(N0, device='cuda', dtype=torch.float32)
inv_d = 1.0 / variances[:, 1]
inv_e = torch.zeros(N0, device='cuda', dtype=torch.float32)
inv_f = 1.0 / variances[:, 2]
inv_cov6 = torch.stack([inv_a, inv_b, inv_c, inv_d, inv_e, inv_f], dim=1)

# Random opacities in [0, 1]
opacities = torch.rand(N0, device='cuda', dtype=torch.float32)

# Call the extract_fields kernel
occ3d = cuda_kernels.extract_fields_launcher(
    means.contiguous(),
    inv_cov6.contiguous(),
    opacities.contiguous(),
    resolution,
    num_blocks,
    float(relax_ratio)
)

print("\nextract_fields output shape:", occ3d.shape)
print("Sample slice at z=16:")
print(occ3d[:, :, 16])
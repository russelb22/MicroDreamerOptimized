import torch
import cuda_kernels  # This is your newly compiled extension

# Create some dummy input data
N = 1024
xyzs = torch.randn(N, 3, device='cuda', dtype=torch.float32)
covs = torch.randn(N, 6, device='cuda', dtype=torch.float32)
out = torch.empty(N, device='cuda', dtype=torch.float32)

# Call your fused CUDA kernel
cuda_kernels.gaussian_3d_launcher(xyzs.contiguous(), covs.contiguous(), out)

# Check result
print("Output:", out[:10])

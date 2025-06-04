// cuda_kernels.cu

#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------
// Existing gaussian_3d_coeff_gpu
// -----------------------------------------------

__global__ void gaussian_3d_coeff_gpu(
    const float* xyzs,
    const float* covs,
    float* out,
    int          N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Read input values
    float x = xyzs[i * 3 + 0];
    float y = xyzs[i * 3 + 1];
    float z = xyzs[i * 3 + 2];

    float a = covs[i * 6 + 0];
    float b = covs[i * 6 + 1];
    float c = covs[i * 6 + 2];
    float d = covs[i * 6 + 3];
    float e = covs[i * 6 + 4];
    float f = covs[i * 6 + 5];

    // Compute inverse covariance terms
    float det = a * d * f + 2 * e * c * b - e * e * a - c * c * d - b * b * f + 1e-24f;
    float inv_det = 1.0f / det;

    float inv_a = (d * f - e * e) * inv_det;
    float inv_b = (e * c - b * f) * inv_det;
    float inv_c = (e * b - c * d) * inv_det;
    float inv_d = (a * f - c * c) * inv_det;
    float inv_e = (b * c - e * a) * inv_det;
    float inv_f = (a * d - b * b) * inv_det;

    // Compute exponent
    float p = -0.5f * (x * x * inv_a + y * y * inv_d + z * z * inv_f)
        - x * y * inv_b - x * z * inv_c - y * z * inv_e;

    if (p > 0.0f)
        p = -1e10f;

    // Output final value
    out[i] = expf(p);
}

extern "C" void launch_gaussian_3d_coeff_gpu(
    const float* xyzs,
    const float* covs,
    float* out,
    int          N
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gaussian_3d_coeff_gpu << <blocks, threads >> > (xyzs, covs, out, N);
}


// -----------------------------------------------
// New: extract_fields_kernel (one thread per voxel)
// -----------------------------------------------

// Each thread computes one voxel-occupancy by summing contributions from all N0 Gaussians.
// Inputs:
//    d_means    : [N0  3]      (Gaussian centers in normalized [-1,1] space)
//    d_inv_cov6 : [N0  6]      (packed inverse covariance per Gaussian: [inv_a,inv_b,inv_c,inv_d,inv_e,inv_f])
//    d_opacity  : [N0]          (opacity scalar for each Gaussian)
//    N0         : int           (number of active Gaussians)
//    resolution : int           (e.g. 128)
//    num_blocks : int           (e.g. 16)  such that split_size = resolution / num_blocks
//    relax_ratio: float         (e.g. 1.5f)
//    block_size : float         (= 2.0f / num_blocks)
// Output:
//    d_occ      : [resolution^3] (flattened occupancy grid)
__global__ void extract_fields_kernel(
    const float* __restrict__ d_means,      // [N03]
    const float* __restrict__ d_inv_cov6,   // [N06]
    const float* __restrict__ d_opacity,    // [N0]
    int          N0,
    int          resolution,
    int          num_blocks,
    int          split_size,
    float        relax_ratio,
    float        block_size,
    float* __restrict__ d_occ         // [resolution^3]
) {
    // 1) Identify which sub-cube block we are in:
    int bx = blockIdx.x;  // range [0..num_blocks-1]
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // 2) Identify our local thread coordinates inside the block:
    int tx = threadIdx.x; // [0..split_size-1]
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // 3) Compute the global voxel indices (xg, yg, zg):
    int xg = bx * split_size + tx;
    int yg = by * split_size + ty;
    int zg = bz * split_size + tz;

    // 4) Compute world-space coordinate for this voxels center in [-1,1]:
    float fx = -1.0f + (2.0f * xg + 1.0f) / float(resolution);
    float fy = -1.0f + (2.0f * yg + 1.0f) / float(resolution);
    float fz = -1.0f + (2.0f * zg + 1.0f) / float(resolution);

    // 5) Initialize the accumulator to 0.0:
    float accum = 0.0f;

    // 6) Loop over all Gaussians (0 .. N0-1):
    //    We do a quick box-cull first, then compute the 3D Gaussian weight
    //    via the packed inverse covariance (6 elements).
    float span = relax_ratio * block_size;  // half-span for culling
    for (int i = 0; i < N0; ++i) {
        // 6a) Quick culling in each axis:
        float mx = d_means[i * 3 + 0];
        float my = d_means[i * 3 + 1];
        float mz = d_means[i * 3 + 2];
        if (fabsf(fx - mx) > span) continue;
        if (fabsf(fy - my) > span) continue;
        if (fabsf(fz - mz) > span) continue;

        // 6b) Compute delta = p - mean_i:
        float dx = fx - mx;
        float dy = fy - my;
        float dz = fz - mz;

        // 6c) Unpack the six elements:
        float inv_a = d_inv_cov6[i * 6 + 0];
        float inv_b = d_inv_cov6[i * 6 + 1];
        float inv_c = d_inv_cov6[i * 6 + 2];
        float inv_d = d_inv_cov6[i * 6 + 3];
        float inv_e = d_inv_cov6[i * 6 + 4];
        float inv_f = d_inv_cov6[i * 6 + 5];

        // 6d) Compute Mahalanobis-squared: 
        float t0 = dx * inv_a + dy * inv_b + dz * inv_c;
        float t1 = dx * inv_b + dy * inv_d + dz * inv_e;
        float t2 = dx * inv_c + dy * inv_e + dz * inv_f;
        float sq = dx * t0 + dy * t1 + dz * t2;

        // 6e) Compute exponent and clamp if positive:
        float power = -0.5f * sq;
        if (power > 0.0f) continue;  // if positive, treat weight as 0

        float w = __expf(power);
        float o = d_opacity[i];
        accum += o * w;
    }

    // 7) Write the result into the flat occupancy array:
    int idx = (xg * resolution + yg) * resolution + zg;
    d_occ[idx] = accum;
}

// Host-side launcher for extract_fields_kernel:
extern "C" void launch_extract_fields(
    const float* d_means,      // [N03]
    const float* d_inv_cov6,   // [N06]
    const float* d_opacity,    // [N0]
    int          N0,
    int          resolution,
    int          num_blocks,
    float        relax_ratio,
    float* d_occ         // [resolution^3], already allocated
) {
    // Compute split_size and block_size exactly as in Python:
    int split_size = resolution / num_blocks;
    float block_size = 2.0f / float(num_blocks);

    // 3D grid of (num_blocks num_blocks num_blocks):
    dim3 gridDim(num_blocks, num_blocks, num_blocks);
    // Each block has (split_size split_size split_size) threads:
    dim3 blockDim(split_size, split_size, split_size);

    // Launch the kernel:
    extract_fields_kernel << <gridDim, blockDim >> > (
        d_means,
        d_inv_cov6,
        d_opacity,
        N0,
        resolution,
        num_blocks,
        split_size,
        relax_ratio,
        block_size,
        d_occ
        );
}
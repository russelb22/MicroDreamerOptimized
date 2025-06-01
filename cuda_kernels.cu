#include <cuda_runtime.h>
#include <math.h>

__global__ void gaussian_3d_coeff_gpu(
    const float* xyzs,
    const float* covs,
    float* out,
    int N
);

__global__ void gaussian_3d_coeff_gpu(
    const float* xyzs, const float* covs, float* out, int N
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
    int N
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gaussian_3d_coeff_gpu << <blocks, threads >> > (xyzs, covs, out, N);
}
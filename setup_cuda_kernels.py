from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_kernels',
    ext_modules=[
        CUDAExtension(
            name='cuda_kernels',
            sources=['cuda_kernels_wrapper.cpp', 'cuda_kernels.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
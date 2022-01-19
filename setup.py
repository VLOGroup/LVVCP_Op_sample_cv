from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import load, library_paths, BuildExtension
# from cpp_extension import BuildExtension, CUDAExtension, library_paths
from src.custom_CUDA_Extension import custom_CUDAExtension

import os
import torch
import re
import warnings
import subprocess
import time

pytorch_cuda_version = torch.version.cuda
sp = subprocess.run(["nvcc","--version"], capture_output=True)
msg = sp.stdout.decode("utf-8")
nvcc_version = re.search("release ([0-9.]*), ", msg)
nvcc_version = nvcc_version.groups()[0]

# Check if C++11 ABI  was used (True/False)
print("Checking C++11 ABI of your pytorch build... _GLIBCXX_USE_CXX11_ABI")
if torch._C._GLIBCXX_USE_CXX11_ABI:
    CXX11_ABI_Flags="-D_GLIBCXX_USE_CXX11_ABI=1"
else:
    CXX11_ABI_Flags="-D_GLIBCXX_USE_CXX11_ABI=0"
print(f"Using {CXX11_ABI_Flags}")




if nvcc_version != pytorch_cuda_version:
    warnings.warn(f"Current CUDA version does not match the Version was built with! \n"\
                  f"Pytorch:{pytorch_cuda_version}, != NVCC:{nvcc_version}")
    time.sleep(3.0)

setup(
    name='sample_cv_v1',
    version='1.0',
    author="Markus Hofinger",
    author_email="markus.hofinger@icg.tugraz.at",

    description='A Pytorch operator that makes a cost volume using a sampling offset',
    include_dirs=["src/include"],

    # Requirements
    #setup_requires=["setuptools_scm"],
    python_requires=">=3, <4",

    # Versioning
    use_scm_version={"root": ".", "relative_to": __file__, "write_to": "include/_version.py"},


    packages=["sample_cv_op"],
    ext_modules=[
        custom_CUDAExtension(
            name='sample_cv_op.libsamplecv_op',
            sources=['src/sample_cv_cuda.cu',
                     'src/sample_cv.cpp',
                    ],

            is_python_module=False, # Turn off python libraries etc. => we want to use it as a C++ Torch Script Operator!


            runtime_library_dirs = library_paths(), # Adding this will permanently link the correct paths to operator
            extra_compile_args={
                    'cxx': [
                        # '-g', # for debugging purposes
                        '-DNDEBUG', # deactivate assertions
                        '-std=c++14',  # Choose C++ standard
                        CXX11_ABI_Flags,  # the C++11 ABI flags must match those pytorch was built with
                        # '-v'  # Verbose
                    ],
                    'nvcc': [ 
                        # '-Xcompiler','-rdynamic','-lineinfo',  # for line numbers and variable naames in cuda memchecker  
                        # '-G','-O2' # for debuggin purposes
                        '-DNDEBUG', # deactivate assertions
                        "--expt-extended-lambda",
                        '-std=c++14',  # Choose C++ standard
                        CXX11_ABI_Flags,  # the C++11 ABI flags must match those pytorch was built with
                        # '-v'  # Verbose
                        ] },                 

            # extra_compile_args={'cxx': ['-g'],            # for debuggin purposes
            #                     'nvcc': ['-G','-O2']},
                    ),
    ],
    # data_files=[],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)


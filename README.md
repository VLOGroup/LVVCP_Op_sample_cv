# Sample CV 
### Overview

This library provides a TorchScrip compatible Cost-Volume sample operator with additional offset input.

This simple version assumes integer offsets and will round any floating point offset to the nearest integer.
Also it does not provide a backward operation in its current form and ist intended for inference only.




### Usefull commands for Debugging:
- Building CUDA with CMAKE syntax changed quite a few times.
  - before CMAKE 3.8 custom versions where common
  - with CMAKE 3.8 CUDA became std. macros like FindCUDA, cuda_select_nvcc_arch_flags and CUDA_NVCC_FLAGS became std.
  - with CMAKE3.18 a new std. was introduced and FindCUDA CUDA_NVCC_FLAGS became depricated - now use  CMAKE_CUDA_FLAGS instead and set the project type to CUDA
- Problems with CUDA Architecture / ARCH Flags (simplified):
  - NVCC can generate PTX (virtual intermediate representation/assembly) and SASS (real machine code) code. As PTX is an intermediate representation it can be JIT compiled into SASS machine code also for newer GPU generations but requieres extra startup time for that. Therefore one can generate fatbinaries that already contain PTX and SASS for different architectures at once.
    - Explicitly forcing the build system to use specific CUDA ARCH and CODE flags to be used within TORCHs version of the setuptools. This means this flag is only recognized by (setup.py). Here some examples:<br>
       `export TORCH_CUDA_ARCH_LIST=7.5`  Using more than one parameter seems not to be possible with older cmake versions 
       `export TORCH_CUDA_ARCH_LIST="6.5 7.5"`  Using more than one parameter seems not to be possible with older cmake versions 
       `export TORCH_CUDA_ARCH_LIST=ALL`  
    - Check which flags where used to build your precompiled pytorch:<br>
      `torch.__config__.show()` <br>
      `torch.cuda.get_arch_list()`
    - Investigate the libraries binary file, to see which architecturs PTX/ELF where integrated:<br>
      `cuobjdump <objfilename>`<br
      `cuobjdump <objfilename> -lelf -lptx`
  
  - Seeing calls to g++ and nvcc:
    - with python distutils:<br>
      `python  setup.py --verbose`
    - with cmake : <br>
      `make VERBOSE=1`
   - `CUDA error: no kernel image is available for execution on the device` indicates that the cuda kernel was not built for your graphics card


### Linklist
This demo was built using information of these very good web sources:

[EXTENDING TORCHSCRIPT WITH CUSTOM C++ OPERATORS](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)<br>
[REGISTERING A DISPATCHED OPERATOR IN C++](https://pytorch.org/tutorials/advanced/dispatcher.html)<br>
[Custom C++ Autograd](https://pytorch.org/tutorials/advanced/cpp_autograd)<br>
[(old) Source Code for this tutorial ](https://github.com/pytorch/extension-script/)<br>
[TorchScript intro](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)<br>
[TorchScript Jit inof](https://pytorch.org/docs/stable/jit.html)<br>
[PyTorch C++ API](https://pytorch.org/cppdocs/)<br>
[OptoX - our previous framework - currently not TorchScriptable](https://github.com/VLOGroup/optox)<br>

[Pytorch C10::ArrayRef References](https://pytorch.org/cppdocs/api/classc10_1_1_array_ref.html)
[Pytorch c10::IValue Reference](https://pytorch.org/cppdocs/api/structc10_1_1_i_value.html)
[CUDA NVCC Compiler Docu (PDF)](https://docs.nvidia.com/cuda/archive/10.1/pdf/CUDA_Compiler_Driver_NVCC.pdf)

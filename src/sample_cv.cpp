// #include <torch/extension.h> // old ways

#include <torch/torch.h>
#include <torch/script.h>

#include <ATen/NamedTensorUtils.h>


#include <vector>
#include "include/sample_cv.hpp"


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void initSampleCV()
{
}

torch::Tensor sample_cv_forward(
    const torch::Tensor &f1,
    const torch::Tensor &f2,
    const torch::Tensor &ofs,
    const torch::IValue rx,
    const torch::IValue ry,
    const torch::IValue NCHW_nNHWC,
    const torch::IValue bilinear)
{
  CHECK_INPUT(f1);
  CHECK_INPUT(f2);
  CHECK_INPUT(ofs);

 TORCH_CHECK(rx.isInt(),  "rx must be an integer type")
 TORCH_CHECK(ry.isInt(),  "ry must be an integer type")
 TORCH_CHECK(NCHW_nNHWC.isBool(),  "NCHW_nNHWC must be a boolean type")
 TORCH_CHECK(bilinear.isBool(),  "bilinear must be a boolean type")

  return sample_cv_forward_cuda_interface(f1, f2, ofs, rx.toInt(), ry.toInt(), NCHW_nNHWC.toBool(), bilinear.toBool());
}

// Advanced Way with Argument names and default values
TORCH_LIBRARY(sample_cv_op, m) {
  m.def("sample_cv_forward(Tensor f1, Tensor f2, Tensor ofs, Any rx, Any ry, Any NCHW_nNHWC, Any bilinear) -> Tensor", sample_cv_forward);
}
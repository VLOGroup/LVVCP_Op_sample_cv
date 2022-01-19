#pragma once
#include <torch/script.h>

#if defined(_MSC_VER)
	#ifdef sample_cv_op_EXPORTS
		#define SAMPLE_CV_OP_API __declspec(dllexport)
	#else
		#define SAMPLE_CV_OP_API __declspec(dllimport)
	#endif
#elif defined(__GNUC__)
	#ifdef sample_cv_op_EXPORTS
		#define SAMPLE_CV_OP_API __attribute__((visibility("default")))
	#else
		#define SAMPLE_CV_OP_API
	#endif
#endif

// Since we don't directly use the sample_cv operator library
// the linker would normally discard the library because
// it thinks we don't use it at all. When this happens
// PyTorch won't be able to find the operator. The compilers
// on Linux support the flag "-Wl,--no-as-needed" which prevents
// the library from being discarded. The MSVC doesn't have a flag
// which allows you to do this. To workaround this there is an initPad2d
// dummy function which needs to be called so that the linker doesn't discard the
// library.
SAMPLE_CV_OP_API void initSampleCV();

// CUDA forward declarations
torch::Tensor sample_cv_forward_cuda_interface(
    const torch::Tensor &x1,
    const torch::Tensor &x2,
    const torch::Tensor &ofs,
    const int rx,
    const int ry,
    const bool use_NCHW_internally,
	const bool bilinear);

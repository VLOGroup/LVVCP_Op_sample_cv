// #include <torch/extension.h>


// #include <cuda.h>
// #include <cuda_runtime.h>
#include <vector>

#include <ATen/cuda/detail/IndexUtils.cuh>
#include <c10/cuda/CUDAStream.h>
#include <THC/THCAtomics.cuh> // keeping THC headers for gpuAtomicAdd
#include <torch/script.h> // One-stop header.
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <iostream>


#include "stdio.h"
#include "include/sample_cv.hpp"
#include "include/CUDAException.h"

// for debugging
// #define CUDA_ERROR_CHECK


constexpr int MAX_BLOCK_SIZE = 512;
constexpr int WARP_SIZE = 32; // Typically 32

#define BLOCK_SPATIAL 32
#define BLOCK_NEIGHBOR 16 
#define MAX_CUDA_BLOCK_SIZE 512 
// 512/32 = 16


#define cudaSafeCall( err ) __cnnCudaSafeCall( err, __FILE__, __LINE__ )

inline void __cnnCudaSafeCall( cudaError_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
#endif
  return;
}

// WarpReduceSum => requires at least Kepler cards
// Combines all values of the same variable within a warp
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
// https://developer.nvidia.com/blog/cooperative-groups/
template <int tile_sz, typename T>
__device__ T reduce_sum_tile_shfl(cg::thread_block_tile<tile_sz> g, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    #pragma unroll
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }

    return val; // note: only thread 0 will return full sum
}

// helpers
__forceinline__ __device__ float myabs(const float x)
{
  return fabsf(x);
}

__forceinline__ __device__ double myabs(const double x)
{
  return fabs(x);
}

template <typename index_t>
__forceinline__ __device__ bool is_valid_index(index_t ix, index_t iy, index_t W ,index_t H){
  return  ( (0<=iy) && (iy<H) && (0<=ix) && (ix<W) );
}


// CUDA kernels
template <typename T>
__global__ void sample_cv_cuda_kernel_NHWC(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> f1,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> f2,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> ofs,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> cv,
  int rx, int ry)
{
  // Data is NHWC
  const int H1 = f1.size(1);
  const int W1 = f1.size(2);
  const int H2 = f2.size(1);
  const int W2 = f2.size(2);
  const int C = f1.size(3);
  const int CVx = (2*rx+1); // width of search space
  const int CVy = (2*ry+1); // height of search space
  const int CVsz = CVx*CVy; // size of search space

  // Data is NHWC we generate NHW dy dx  data
  //  Multiple threads run parallel over C and dy dx
  const int ix = blockIdx.x;
  const int iy = blockIdx.y;
  const int in = blockIdx.z;
  
  if(iy < 0 || iy  >= H1 || ix  < 0 || ix >= W1)
    return;

  // Use all Threads to cache the central pixel from global memory to faster local memory
  extern __shared__ char smem_char[];
  T *smem_f1 = reinterpret_cast<T*>(smem_char);
  for (int ic=threadIdx.y * blockDim.x +  threadIdx.x; ic < C; ic += blockDim.x * blockDim.y){
    smem_f1[ic] = f1[in][iy][ix][ic];
  }
  __syncthreads();

  // cache pixel offset
  // neareast neighbour => round flow to integer values if within +/- 0.5 pixel
  const int ofs_x = static_cast<int>( ofs[in][0][iy][ix] + 0.5) ;
  const int ofs_y = static_cast<int>( ofs[in][1][iy][ix] + 0.5) ;


  // see https://developer.nvidia.com/blog/cooperative-groups/
  auto tile32 = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());  // group the threads into tiles of 32 threads each

  // iterate over the search space (is also done in parallel in HW by y-Threads)
  for (int dxy = threadIdx.y; dxy < CVsz; dxy += blockDim.y){
    const int iCVx = dxy % CVx; // Cost Volume index x 
    const int iCVy = dxy / CVx; // Cost Volume index y
    const int idx = ix + iCVx - rx + ofs_x; // Cost Volume offset x (pix)
    const int idy = iy + iCVy - ry + ofs_y; // Cost Volume offset y (pix)

    // Check that the offset position is still within the target image
    if  ( (0<=idy) && (idy<H2) && (0<=idx) && (idx<W2) ) {
      // use x-Threads to load feature channels
      T sum=0;
      for (int ic = threadIdx.x; ic < C; ic +=blockDim.x){
        sum += smem_f1[ic] * f2[in][idy][idx][ic];
        // sum += f1[in][iy][ix][ic] * f2[in][idy][idx][ic];
      }
      
      // Now the sums need to be merged
      // v1) use attomic ad
      // gpuAtomicAdd(&cv[in][iy][ix][iCVy][iCVx], sum);
      
      // v2) use Warpsum
      // Sum up using warp tiles => Every Warp (32Threads) sum together
      // Our 32 x threads operate over the C dimension => sum up C
      // however, other tiles can work on different positions
      // Therefore one thread per tile must write back the resulting sum
      // see https://developer.nvidia.com/blog/cooperative-groups/
      T tile_sum = reduce_sum_tile_shfl<32, T>(tile32, sum);
      if (tile32.thread_rank() == 0){
        cv[in][iy][ix][iCVy][iCVx] = tile_sum;
      }
    }
  }
}



// CUDA kernels
template <typename T>
__global__ void sample_cv_cuda_kernel_bilinear_NHWC(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> f1,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> f2,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> ofs,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> cv,
  int rx, int ry)
{
  // Data is NHWC
  const int H1 = f1.size(1);
  const int W1 = f1.size(2);
  const int H2 = f2.size(1);
  const int W2 = f2.size(2);
  const int C = f1.size(3);
  const int CVx = (2*rx+1); // width of search space
  const int CVy = (2*ry+1); // height of search space
  const int CVsz = CVx*CVy; // size of search space

  // Data is NHWC we generate NHW dy dx  data
  //  Multiple threads run parallel over C and dy dx
  const int ix = blockIdx.x;
  const int iy = blockIdx.y;
  const int in = blockIdx.z;
  
  if(iy < 0 || iy  >= H1 || ix  < 0 || ix >= W1)
    return;

  // Use all Threads to cache the central pixel from global memory to faster local memory
  extern __shared__ char smem_char[];
  T *smem_f1 = reinterpret_cast<T*>(smem_char);
  for (int ic=threadIdx.y * blockDim.x +  threadIdx.x; ic < C; ic += blockDim.x * blockDim.y){
    smem_f1[ic] = f1[in][iy][ix][ic];
  }
  __syncthreads();

  // cache pixel offset
  T x =  ofs[in][0][iy][ix];
  T y =  ofs[in][1][iy][ix];

  const int x_tl = static_cast<int> ( floor(x)); 
  const int y_tl = static_cast<int> ( floor(y)); 

  const int x_br = x_tl +1;
  const int y_br = y_tl +1;
  const int x_tr = x_tl +1;
  const int y_tr = y_tl;
  const int x_bl = x_tl;
  const int y_bl = y_tl+1;
  
  T A_tl = (x_br - x) * (y_br - y);
  T A_tr = (x - x_bl) * (y_bl - y);
  T A_bl = (x_tr - x) * (y - y_tr);
  T A_br = (x - x_tl) * (y - y_tl);


  // see https://developer.nvidia.com/blog/cooperative-groups/
  auto tile32 = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());  // group the threads into tiles of 32 threads each

  // iterate over the search space (is also done in parallel in HW by y-Threads)
  for (int dxy = threadIdx.y; dxy < CVsz; dxy += blockDim.y){
    const int iCVx = dxy % CVx; // Cost Volume index x 
    const int iCVy = dxy / CVx; // Cost Volume index y
    const int idx = ix + iCVx - rx ; // Cost Volume offset x (pix)
    const int idy = iy + iCVy - ry ; // Cost Volume offset y (pix)

    // Check that the offset position is still within the target image
    // if  ( (0<=idy) && (idy<H2) && (0<=idx) && (idx<W2) ) {
    if (true){ // alternatively check each of the 4 pixels individually
      // use x-Threads to load feature channels
      T sum=0;
      if  ( is_valid_index(idx + x_tl, idy + y_tl, W2, H2)) {
        for (int ic = threadIdx.x; ic < C; ic +=blockDim.x){
        sum += smem_f1[ic] * f2[in][idy + y_tl][idx + x_tl][ic] * A_tl;
        }
      }
      if  ( is_valid_index(idx + x_tr, idy + y_tr, W2, H2)) {
        for (int ic = threadIdx.x; ic < C; ic +=blockDim.x){
        sum += smem_f1[ic] * f2[in][idy + y_tr][idx + x_tr][ic] * A_tr;
        }
      }
      if  ( is_valid_index(idx + x_bl, idy + y_bl, W2, H2)) {
        for (int ic = threadIdx.x; ic < C; ic +=blockDim.x){
        sum += smem_f1[ic] * f2[in][idy + y_bl][idx + x_bl][ic] * A_bl;
        }
      }      
      if  ( is_valid_index(idx + x_br, idy + y_br, W2, H2)) {
        for (int ic = threadIdx.x; ic < C; ic +=blockDim.x){
        sum += smem_f1[ic] * f2[in][idy + y_br][idx + x_br][ic] * A_br;
        }
      }
      
      // Now the sums need to be merged
      // v1) use attomic ad
      // gpuAtomicAdd(&cv[in][iy][ix][iCVy][iCVx], sum);
      
      // v2) use Warpsum
      // Sum up using warp tiles => Every Warp (32Threads) sum together
      // Our 32 x threads operate over the C dimension => sum up C
      // however, other tiles can work on different positions
      // Therefore one thread per tile must write back the resulting sum
      // see https://developer.nvidia.com/blog/cooperative-groups/
      T tile_sum = reduce_sum_tile_shfl<32, T>(tile32, sum);
      if (tile32.thread_rank() == 0){
        cv[in][iy][ix][iCVy][iCVx] = tile_sum;
      }
    }
  }
}


// CUDA kernels
template <typename T>
__global__ void sample_cv_cuda_kernel_NCHW(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> f1,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> f2,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> ofs,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> cv,
  int rx, int ry)
{
  const int ixy = blockIdx.x * blockDim.x + threadIdx.x;
  const int iuv = blockIdx.y * blockDim.y + threadIdx.y;
  const int ib   = blockIdx.z * blockDim.z + threadIdx.z;

  const int H1 = f1.size(2); // [N,C,H,W]
  const int W1 = f1.size(3);
  const int H2 = f2.size(2);
  const int W2 = f2.size(3);
  const int C  = f1.size(1);
  const int CVx = (2*rx+1); // width of search space
  const int CVy = (2*ry+1); // width of search space

  const int ix = ixy % W1;
  const int iy = ixy / W1;

  // Our tiles can be bigger than the image
  if ( (iuv >= CVx*CVy) || (ixy >= H1*W1) )
    return;

  if(iy < 0 || iy  >= H1 || ix  < 0 || ix >= W1)
    return;

  // neareast neighbour => round flow to integer values if within +/- 0.5 pixel
  const int ofs_x = static_cast<int>( ofs[ib][0][iy][ix] + 0.5) ;
  const int ofs_y = static_cast<int>( ofs[ib][1][iy][ix] + 0.5) ;

  const int iu = iuv % CVx;  // index position in cost volume
  const int iv = iuv / CVx;
  const int ix2 = ix + ofs_x  + (iu - rx);
  const int iy2 = iy + ofs_y  + (iv - ry);

  // check inside f2
  if( (iy2 < 0) || (iy2 >= H2) || (ix2 < 0) || (ix2 >= W2))
    return;

  T sim = 0;
  for(int ic = 0; ic < C; ++ic)
  { // cosine distance
    sim += f1[ib][ic][iy][ix] * f2[ib][ic][iy2][ix2];
  }
  cv[ib][iy][ix][iv][iu] = sim;
}


// CUDA kernels
template <typename T>
__global__ void sample_cv_cuda_kernel_bilinear_NCHW(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> f1,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> f2,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> ofs,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> cv,
  int rx, int ry)
{
  const int ixy = blockIdx.x * blockDim.x + threadIdx.x;
  const int idxy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ib   = blockIdx.z * blockDim.z + threadIdx.z;

  const int H1 = f1.size(2); // [N,C,H,W]
  const int W1 = f1.size(3);
  const int H2 = f2.size(2);
  const int W2 = f2.size(3);
  const int C  = f1.size(1);
  const int CVx = (2*rx+1); // width of search space
  const int CVy = (2*ry+1); // width of search space

  const int ix = ixy % W1;
  const int iy = ixy / W1;

  // Our tiles can be bigger than the image
  if ( (idxy >= CVx*CVy) || (ixy >= H1*W1) )
    return;

  if(iy < 0 || iy  >= H1 || ix  < 0 || ix >= W1)
    return;

  // cache pixel offset
  const T x =  ofs[ib][0][iy][ix];
  const T y =  ofs[ib][1][iy][ix];

  const int x_tl = static_cast<int> ( floor(x)); 
  const int y_tl = static_cast<int> ( floor(y)); 

  const int x_br = x_tl +1;
  const int y_br = y_tl +1;
  const int x_tr = x_tl +1;
  const int y_tr = y_tl;
  const int x_bl = x_tl;
  const int y_bl = y_tl+1;
  
  const T A_tl = (x_br - x) * (y_br - y);
  const T A_tr = (x - x_bl) * (y_bl - y);
  const T A_bl = (x_tr - x) * (y - y_tr);
  const T A_br = (x - x_tl) * (y - y_tl);

  const int iCVx = idxy % CVx; // Cost Volume index x 
  const int iCVy = idxy / CVx; // Cost Volume index y
  const int idx = ix + (iCVx - rx); // search position in 2nd image without offset
  const int idy = iy + (iCVy - ry); // search position in 2nd image without offset

  T sim = 0;
  for(int ic = 0; ic < C; ++ic)
  { // cosine distance
    const T f1_tmp =  f1[ib][ic][iy][ix];
    if ( is_valid_index(idx + x_tl, idy + y_tl, W2, H2)) {
      sim += f1_tmp * f2[ib][ic][idy + y_tl][idx + x_tl] * A_tl;
    }
    if  ( is_valid_index(idx + x_tr, idy + y_tr, W2, H2)) {
      sim += f1_tmp * f2[ib][ic][idy + y_tr][idx + x_tr] * A_tr;
    }
    if  ( is_valid_index(idx + x_bl, idy + y_bl, W2, H2)) {
      sim += f1_tmp * f2[ib][ic][idy + y_bl][idx + x_bl] * A_bl;
    }      
    if  ( is_valid_index(idx + x_br, idy + y_br, W2, H2)) {
      sim += f1_tmp * f2[ib][ic][idy + y_br][idx + x_br] * A_br;
    }
  }

  cv[ib][iy][ix][iCVy][iCVx] = sim;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface for C++ to launch cuda kernel
torch::Tensor sample_cv_forward_cuda_interface(
  const torch::Tensor &f1,
  const torch::Tensor &f2,
  const torch::Tensor &ofs,
  const int rx,
  const int ry,
  const bool use_NCHW_internally,
  const bool bilinear)
{
  TORCH_CHECK(f1.dim() == 4, "Expected 4d tensor, but got");
  TORCH_CHECK(f2.dim() == 4, "Expected 4d tensor, but got");
  TORCH_CHECK(f1.size(0) == f2.size(0), "Both inputs must have same number of batches");
  TORCH_CHECK(f1.size(1) == f2.size(1), "Both inputs must have same number of channels");

  // [N,C,H,W]
  const int B = f1.size(0);
  const int C = f1.size(1);
  const int H = f1.size(2);
  const int W = f1.size(3);
  const int CVsz = (2*rx+1)*(2*ry+1);

  // auto dist = torch::ones({B, H, W, 2*ry+1, 2*rx+1}, f1.options()) * 100; // for debugging => find errors
  auto dist = torch::zeros({B, H, W, 2*ry+1, 2*rx+1}, f1.options()) ;

  if (use_NCHW_internally){ // NCHW
    const dim3 threads(BLOCK_SPATIAL, BLOCK_NEIGHBOR, 1);
    const dim3 blocks((H*W+threads.x-1) / threads.x,                 // threads x => all pixels floor division
                        ((2*ry+1)*(2*rx+1)+threads.y-1) / threads.y, // threads y => the cost volume search direction
                        (B+threads.z-1) / threads.z);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (bilinear){
        AT_DISPATCH_FLOATING_TYPES(f1.type(), "sample_cv", ([&]{
          sample_cv_cuda_kernel_bilinear_NCHW<scalar_t><<<blocks, threads, 0, stream>>>(
            f1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            f2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            ofs.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), 
            dist.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(), 
            rx, ry);
        }));
    }else{
        AT_DISPATCH_FLOATING_TYPES(f1.type(), "sample_cv", ([&]{
          sample_cv_cuda_kernel_NCHW<scalar_t><<<blocks, threads, 0, stream>>>(
            f1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            f2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            ofs.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), 
            dist.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(), 
            rx, ry);
        }));
    }
  }else{ 
    // use NHWC internally => reorder input data (which is NCHW)
    auto x1p = f1.permute({0,2,3,1}); // [NCHW]=>[NHWC]
    auto x2p = f2.permute({0,2,3,1}); // [NCHW]=>[NHWC]
    x1p = x1p.contiguous();
    x2p = x2p.contiguous();
    // each thread block computes the complete search window for one pixel
    const dim3 threads(WARP_SIZE, min(CVsz, MAX_BLOCK_SIZE/WARP_SIZE), 1);  
    // const dim3 threads(WARP_SIZE, 1, 1);  
    const dim3 blocks( W,H,B); // repeat for all pixels
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (bilinear){
      AT_DISPATCH_FLOATING_TYPES(f1.type(), "sample_cv", ([&]{
        sample_cv_cuda_kernel_bilinear_NHWC<scalar_t><<<blocks, threads,  C * sizeof(scalar_t), stream>>>(
        // sample_cv_cuda_kernel_NHWC<scalar_t><<<blocks, threads,  C * sizeof(scalar_t), stream>>>(
          x1p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          x2p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          ofs.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), 
          dist.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(), 
          rx, ry);
      }));
    }else{
      AT_DISPATCH_FLOATING_TYPES(f1.type(), "sample_cv", ([&]{
        sample_cv_cuda_kernel_NHWC<scalar_t><<<blocks, threads,  C * sizeof(scalar_t), stream>>>(
        // sample_cv_cuda_kernel_NHWC<scalar_t><<<blocks, threads,  C * sizeof(scalar_t), stream>>>(
          x1p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          x2p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          ofs.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), 
          dist.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(), 
          rx, ry);
      }));
    }

  }

  // cudaSafeCall(cudaGetLastError());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return dist;
}

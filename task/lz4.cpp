//
// Created by 1 on 14.02.2022.
//
#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"

#include <stdlib.h>
#include <vector>

// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return 0;                                                                \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)



#include "test_data.h"

int main()
{

  using T = uint8_t;
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t comp_temp_bytes = 0;
  size_t comp_out_bytes = 0;
  void* d_comp_temp;
  void* d_comp_out;

  std::vector<size_t> chunk_sizes{
      32768, 32769, 50000, 65535, 65536, 90103, 16777216};

  const size_t chunk_size = chunk_sizes[1];

  LZ4Compressor compressor(chunk_size, NVCOMP_TYPE_UCHAR);
  compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);
  REQUIRE(comp_temp_bytes > 0);
  REQUIRE(comp_out_bytes > 0);

  // allocate temp buffer
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Allocate output buffer
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  size_t* comp_out_bytes_ptr;
  cudaMalloc((void**)&comp_out_bytes_ptr, sizeof(size_t));
  compressor.compress_async(
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes_ptr,
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(
      &comp_out_bytes,
      comp_out_bytes_ptr,
      sizeof(comp_out_bytes),
      cudaMemcpyDeviceToHost));
  cudaFree(comp_out_bytes_ptr);

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  // Test to make sure copying the compressed file is ok
  void* copied = 0;
  CUDA_CHECK(cudaMalloc(&copied, comp_out_bytes));
  CUDA_CHECK(
      cudaMemcpy(copied, d_comp_out, comp_out_bytes, cudaMemcpyDeviceToDevice));
  cudaFree(d_comp_out);
  d_comp_out = copied;

  printf("input bytes : %u\n", in_bytes);
  printf("compress bytes : %u\n", comp_out_bytes);

  printf("\ndone\n");
}
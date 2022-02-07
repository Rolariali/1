#include <random>
#include <assert.h>
#include <iostream>

#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

// #include "catch.hpp"
#include <assert.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace nvcomp;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

int main()
{
  using T = int;

  int packing = 0;
  int RLE = 1;
  int Delta = 1;
  std::vector<T> input = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};
  size_t chunk_size = 10000;

  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  (cudaMalloc((void**)&d_in_data, in_bytes));
  (
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t comp_temp_bytes = 0;
  size_t comp_out_bytes = 0;
  void* d_comp_temp;
  void* d_comp_out;

  // Get comptess temp size
  CascadedCompressor compressor(TypeOf<T>(), RLE, Delta, packing);

  compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);
  (comp_temp_bytes > 0);
  (comp_out_bytes > 0);

  std::cout << "comp_temp_bytes" << comp_temp_bytes << std::endl;
  std::cout << "comp_out_bytes" << comp_out_bytes << std::endl;


  // allocate temp buffer
  (cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Allocate output buffer
  (cudaMalloc(&d_comp_out, comp_out_bytes));

  size_t* comp_out_bytes_ptr;
  cudaMalloc((void**)&comp_out_bytes_ptr, sizeof(*comp_out_bytes_ptr));
  compressor.compress_async(
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes_ptr,
      stream);

  (cudaStreamSynchronize(stream));
  (cudaMemcpy(
      &comp_out_bytes,
      comp_out_bytes_ptr,
      sizeof(comp_out_bytes),
      cudaMemcpyDeviceToHost));
  cudaFree(comp_out_bytes_ptr);

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  size_t temp_bytes = 0;
  size_t num_out_bytes = 0;
  void* temp_ptr;
  T* out_ptr;

  std::cout << "test compression " << comp_out_bytes << std::endl;

  return 0;

}
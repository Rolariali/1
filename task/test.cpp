#include <random>
#include <assert.h>
#include <iostream>
#include <iomanip>


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
    assert(err == cudaSuccess);                                               \
  } while (false)

int main()
{
  using T = uint8_t;

  int packing = 0;
  int RLE = 1;
  int Delta = 1;
  std::vector<T> input = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};
  size_t chunk_size = 10000;

  std::cout << "size input  " << input.size() << std::endl;
  // std::cout << "sze compression " << comp_out_bytes << std::endl;
  for (uint8_t i: input)
    cout << std::setfill ('0') << std::setw(sizeof(uint8_t)*2)
         << std::hex << (int)i ;
  cout << endl;

  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  std::cout << "in_bytes qty: " << in_bytes << std::endl;
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::cout << "------------------------- Copress --------------------" << std::endl;

  size_t comp_temp_bytes = 0;
  size_t comp_out_bytes = 0;
  void* d_comp_temp;
  void* d_comp_out;

  // Get comptess temp size
  CascadedCompressor compressor(TypeOf<T>(), RLE, Delta, packing);

  compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);
  assert(comp_temp_bytes > 0);
  assert(comp_out_bytes > 0);

  std::cout << "comp_temp_bytes " << comp_temp_bytes << std::endl;
  std::cout << "comp_out_bytes " << comp_out_bytes << std::endl;

  // allocate temp buffer
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Allocate output buffer
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));
  // do
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
  // wait
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // copy from
  CUDA_CHECK(cudaMemcpy(
      &comp_out_bytes,
      comp_out_bytes_ptr,
      sizeof(comp_out_bytes),
      cudaMemcpyDeviceToHost));
  // free
  cudaFree(comp_out_bytes_ptr);
  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  vector<uint8_t> compres_data(comp_out_bytes, 12);
  CUDA_CHECK(cudaMemcpy(
      &compres_data[0],
      d_comp_out,
      comp_out_bytes,
      cudaMemcpyDeviceToHost));

  std::cout << "size compression  " << comp_out_bytes << std::endl;
//  std::cout << "n " << std::endl;
  for (uint8_t i: compres_data)
    cout << std::setfill ('0') << std::setw(sizeof(uint8_t)*2)
         << std::hex << (int)i ;
//  std::cout << compres_data << std::endl;
//  std::cout << ((uint32_t*)d_comp_out)[0] << std::endl;
//  for (int i=0; i < 5; i++)
//    std::cout << std::hex << ((T*)d_comp_out)[i];

  std::cout << std::endl;

std::cout << "------------------------- Decopress --------------------" << std::endl;

  size_t temp_bytes = 0;
  size_t num_out_bytes = 0;
  void* temp_ptr;
  T* out_ptr;

  CascadedDecompressor decompressor;

  // get temp size
  decompressor.configure(
      d_comp_out, comp_out_bytes, &temp_bytes, &num_out_bytes, stream);
  assert(temp_bytes > 0);
  assert(num_out_bytes == in_bytes);

  // allocate temp buffer
  cudaMalloc(&temp_ptr, temp_bytes); // also can use RMM_ALLOC instead

  // allocate output buffer
  cudaMalloc(&out_ptr, num_out_bytes); // also can use RMM_ALLOC instead

  // execute decompression (asynchronous)
  decompressor.decompress_async(
      d_comp_out,
      comp_out_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      num_out_bytes,
      stream);

  cudaStreamSynchronize(stream);

  // Copy result back to host
  std::vector<T> res(num_out_bytes / sizeof(T));
  cudaMemcpy(&res[0], out_ptr, num_out_bytes, cudaMemcpyDeviceToHost);

  cudaFree(temp_ptr);
  cudaFree(d_comp_out);
  cudaFree(out_ptr);

  // Verify correctness
  assert(res == input);

  std::cout << "size decompression " << num_out_bytes << std::endl;
  // std::cout << "sze compression " << comp_out_bytes << std::endl;
  for (uint8_t i: res)
    cout << std::setfill ('0') << std::setw(sizeof(uint8_t)*2)
         << std::hex << (int)i ;


  return 0;

}
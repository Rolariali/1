//
// Created by 1 on 25.02.2022.
//

//
// Created by 1 on 25.02.2022.
//

#include "nvcomp.h"
#include "nvcomp/cascaded.h"
#include "../src/highlevel/CascadedMetadata.h"
//#include "cuda_runtime.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
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

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      printf(                                                                  \
          "API call failure \"" #func "\" with %d at " __FILE__ ":%d\n",       \
          (int)rt,                                                             \
          __LINE__);                                                           \
      return 0;                                                                \
    }                                                                          \
  } while (0)


int main()
{
  typedef int8_t T;
  const nvcompType_t type = NVCOMP_TYPE_UCHAR;

  const nvcompCascadedFormatOpts comp_opts ={0,1,0};

  const size_t input_size = 16;
  T input[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};

  // create GPU only input buffer
  void* d_in_data;
  const size_t in_bytes = sizeof(T) * input_size;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompStatus_t status;

  // Compress on the GPU
  size_t comp_temp_bytes;
  size_t comp_out_bytes;
  size_t metadata_bytes;
  status = nvcompCascadedCompressConfigure(
      &comp_opts,
      type,
      in_bytes,
      &metadata_bytes,
      &comp_temp_bytes,
      &comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  size_t* d_comp_out_bytes;
  CUDA_CHECK(cudaMalloc((void**)&d_comp_out_bytes, sizeof(*d_comp_out_bytes)));
  CUDA_CHECK(cudaMemcpy(
      d_comp_out_bytes,
      &comp_out_bytes,
      sizeof(*d_comp_out_bytes),
      cudaMemcpyHostToDevice));

  status = nvcompCascadedCompressAsync(
      &comp_opts,
      type,
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      d_comp_out_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMemcpy(
      &comp_out_bytes,
      d_comp_out_bytes,
      sizeof(comp_out_bytes),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_comp_out_bytes));
  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_in_data));

  std:vector<T> out(comp_out_bytes);
  CUDA_CHECK(cudaMemcpy(
      out.data(), d_comp_out, comp_out_bytes, cudaMemcpyDeviceToHost));


  // get temp and output size
  size_t temp_bytes;
  size_t output_bytes;
  void* metadata_ptr = NULL;

  status = nvcompDecompressGetMetadata(
      d_comp_out, comp_out_bytes, &metadata_ptr, stream);
  REQUIRE(status == nvcompSuccess);
  nvcomp::highlevel::CascadedMetadata* meta_ptr = static_cast<nvcomp::highlevel::CascadedMetadata*>(metadata_ptr);
  size_t meta_bytes = meta_ptr->getDataOffset(0);
  printf("\n==== compress data: ");
  for(int i = meta_bytes;  i < comp_out_bytes; i++)
    printf("%d:", out[i]);

  status = nvcompCascadedDecompressConfigure(
      d_comp_out,
      comp_out_bytes,
      &metadata_ptr,
      &metadata_bytes,
      &temp_bytes,
      &output_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);

  // allocate temp buffer
  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));
  // allocate output buffer
  void* out_ptr;
  CUDA_CHECK(cudaMalloc(&out_ptr, output_bytes));

  // execute decompression (asynchronous)
  status = nvcompCascadedDecompressAsync(
      d_comp_out,
      comp_out_bytes,
      metadata_ptr,
      metadata_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      output_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaDeviceSynchronize());

  nvcompCascadedDestroyMetadata(metadata_ptr);

  // Copy result back to host
  T res[16];
  cudaMemcpy(res, out_ptr, output_bytes, cudaMemcpyDeviceToHost);

  CUDA_CHECK(cudaFree(temp_ptr));
  CUDA_CHECK(cudaFree(d_comp_out));

  // Verify correctness
  printf("\n==== decompress data: ");
  for (size_t i = 0; i < input_size; ++i) {
    printf("%d:", res[i]);
    REQUIRE(res[i] == input[i]);
  }

  return 0;

}
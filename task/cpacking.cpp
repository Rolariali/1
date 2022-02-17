
#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

//#include "highlevel/BitcompMetadata.h"
//#include "highlevel/CascadedMetadata.h"
#include "../src/highlevel/CascadedMetadata.h"
//#include "highlevel/Metadata.h"
// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return nvcompErrorInternal;                                                                \
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
      return false;                                                                \
    }                                                                          \
  } while (0)

struct CompResult{
  vector<uint8_t> output;
  vector<uint8_t> meta;
  size_t out_bytes;
  size_t meta_bytes;

};

bool cascade(vector<uint8_t> input, CompResult & res,
                       int RLE, int deltas, int use_bp){

  typedef uint8_t T;
  const nvcompType_t type = NVCOMP_TYPE_UCHAR;

  const size_t input_size = input.size();
  nvcompCascadedFormatOpts comp_opts;

  comp_opts.num_RLEs = 0;
  comp_opts.num_deltas = 1;
  comp_opts.use_bp = 0;
  // create GPU only input buffer
  void* d_in_data;
  const size_t in_bytes = sizeof(T) * input_size;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(
      d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

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
  CUDA_CHECK(
      cudaMalloc((void**)&d_comp_out_bytes, sizeof(*d_comp_out_bytes)));
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


  vector<uint8_t> out(comp_out_bytes);

  CUDA_CHECK(cudaMemcpy(
      out.data(), d_comp_out, comp_out_bytes, cudaMemcpyDeviceToHost));

  // get temp and output size
  size_t temp_bytes;
  size_t output_bytes;
  void* metadata_ptr = NULL;

  status = nvcompDecompressGetMetadata(
      d_comp_out, comp_out_bytes, &metadata_ptr, stream);
  REQUIRE(status == nvcompSuccess);

  nvcomp::highlevel::CascadedMetadata* m
      = static_cast<nvcomp::highlevel::CascadedMetadata*>(metadata_ptr);


  res.meta_bytes = m->getDataOffset(0);

  res.out_bytes = comp_out_bytes - res.meta_bytes;
  res.output.resize(res.out_bytes);

   for(int i = res.meta_bytes;  i < comp_out_bytes; i++)
       res.output.push_back(out[i]);

   return true;
}


int main()
{
  printf("--------------------------------------------\n");
  vector<uint8_t> input = {3,3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9};
  struct CompResult res;
  REQUIRE(cascade(input, res, 1, 0, 0));

  printf("--------------------------------------------\n");
  printf("\n\ndone\n");
}

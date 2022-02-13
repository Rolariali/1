//
// Created by 1 on 13.02.2022.
//

//#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

// Test GPU decompression with cascaded compression API //

using namespace std;
//using namespace nvcomp;

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



/// Run C API selector with NULL options (default) and return the estimated compression ratio
template <typename T>
double test_selector_default_c(const std::vector<T>& input, nvcompCascadedFormatOpts* opts)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  size_t temp_bytes = 0;
  void* d_temp;

  nvcompStatus_t err = nvcompCascadedSelectorConfigure(
      NULL, nvcomp::TypeOf<T>(), in_bytes, &temp_bytes);
  REQUIRE(err == nvcompSuccess);

  CUDA_CHECK( cudaMalloc(&d_temp, temp_bytes) );

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  double est_ratio;

  err = nvcompCascadedSelectorRun(
      NULL,
      nvcomp::TypeOf<T>(),
      d_in_data,
      in_bytes,
      d_temp,
      temp_bytes,
      opts,
      &est_ratio,
      stream);

  cudaStreamSynchronize(stream);
  REQUIRE(err == nvcompSuccess);

  cudaFree(d_temp);
  cudaFree(d_in_data);

  return est_ratio;
}

#include "test_data.h"

int main()
{

  nvcompCascadedFormatOpts opts;
  typedef uint8_t T;
  double est_ratio = test_selector_default_c<T>(input, &opts);

}
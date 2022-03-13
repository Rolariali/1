
#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define REQUIRE(a) assert(a)                                                     \

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = (cond);                                                  \
    assert(err == cudaSuccess);                                               \
  } while (false)



template <typename T>
void test_cascaded(const std::vector<T>& input, nvcompBatchedCascadedOpts_t options)
{
  // create GPU only input buffer
  T* d_in_data;
  nvcompType_t data_type = nvcomp::TypeOf<T>();
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

//  nvcompBatchedCascadedOpts_t options = nvcompBatchedCascadedDefaultOpts;
  options.type = data_type;
  CascadedManager manager{options, stream};
  auto comp_config = manager.configure_compression(in_bytes);

  // Allocate output buffer
  uint8_t* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

  manager.compress(
      reinterpret_cast<const uint8_t*>(d_in_data),
      d_comp_out,
      comp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t comp_out_bytes = manager.get_compressed_output_size(d_comp_out);
  printf("comp_out_bytes: %zu\n", comp_out_bytes);

  std::vector<uint8_t> out(comp_out_bytes);
  cudaMemcpy(
      &out[0], d_comp_out, comp_out_bytes, cudaMemcpyDeviceToHost);

  for(auto el: out)
    printf("%d:", (int)el);
  printf("\n");
  cudaFree(d_in_data);

  // Test to make sure copying the compressed file is ok
  uint8_t* copied = 0;
  CUDA_CHECK(cudaMalloc(&copied, comp_out_bytes));
  CUDA_CHECK(
      cudaMemcpy(copied, d_comp_out, comp_out_bytes, cudaMemcpyDeviceToDevice));
  cudaFree(d_comp_out);
  d_comp_out = copied;

  auto decomp_config = manager.configure_decompression(d_comp_out);

  T* out_ptr;
  cudaMalloc(&out_ptr, decomp_config.decomp_data_size);

  // make sure the data won't match input if not written to, so we can verify
  // correctness
  cudaMemset(out_ptr, 0, decomp_config.decomp_data_size);

  manager.decompress(
      reinterpret_cast<uint8_t*>(out_ptr),
      d_comp_out,
      decomp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy result back to host
  std::vector<T> res(input.size());
  cudaMemcpy(
      &res[0], out_ptr, input.size() * sizeof(T), cudaMemcpyDeviceToHost);

  // Verify correctness
  REQUIRE(res == input);

  cudaFree(d_comp_out);
  cudaFree(out_ptr);
}

int main()
{
  typedef uint8_t T;

  nvcompBatchedCascadedOpts_t options = nvcompBatchedCascadedDefaultOpts;
  options.num_deltas = 1;
  options.num_RLEs = 0;
  options.use_bp = 0;
//  options.chunk_size = 1;
  std::vector<T> input;

#if 1
  for(int i=0; i<32; i++)
    input.push_back(i);
#else
  for(int i=32; i>0; i--)
    input.push_back(i);
  input.push_back(2);
#endif
  for(auto el: input)
    printf("%d:", el);
  printf("\n");
  test_cascaded<T>(input, options);
}
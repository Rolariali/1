#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <iostream>

using namespace std;
using namespace nvcomp;

#define REQUIRE(a) assert(a)

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

size_t max_compressed_size(size_t uncompressed_size)
{
  return (uncompressed_size + 3) / 4 * 4 + 4;
}

bool print_diff = false;
bool one_only = false;
bool verbose = false;
/**
 * Verify the number of decompressed bytes match the number of the uncompressed
 * bytes.
 */
void verify_decompressed_sizes(
    size_t batch_size,
    const size_t* decompressed_bytes_device,
    const std::vector<size_t>& uncompressed_bytes_host)
{
  std::vector<size_t> decompressed_bytes_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
      decompressed_bytes_host.data(),
      decompressed_bytes_device,
      sizeof(size_t) * batch_size,
      cudaMemcpyDeviceToHost));

  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    REQUIRE(
        decompressed_bytes_host[partition_idx]
        == uncompressed_bytes_host[partition_idx]);
  }
}

/**
 * Verify decompression outputs match the original uncompressed data.
 */
template <typename data_type>
void verify_decompressed_output(
    size_t batch_size,
    const std::vector<void*>& decompressed_ptrs_host,
    const std::vector<const data_type*>& uncompressed_data_host,
    const std::vector<size_t>& uncompressed_bytes_host)
{

  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    const size_t num_elements
        = uncompressed_bytes_host[partition_idx] / sizeof(data_type);

    std::vector<data_type> decompressed_data_host(num_elements);
    CUDA_CHECK(cudaMemcpy(
        decompressed_data_host.data(),
        decompressed_ptrs_host[partition_idx],
        uncompressed_bytes_host[partition_idx],
        cudaMemcpyDeviceToHost));
    if(print_diff)
      printf("\nverify:\n");
    for (size_t element_idx = 0; element_idx < num_elements; element_idx++) {
      if(print_diff)
        printf("%u\t%d == %d\n", element_idx,
               decompressed_data_host[element_idx],
                                uncompressed_data_host[partition_idx][element_idx]);
      REQUIRE(
          decompressed_data_host[element_idx]
          == uncompressed_data_host[partition_idx][element_idx]);
    }
  }
}


template <typename T>
size_t test_cascaded(const std::vector<T>& input,
                     const nvcompBatchedCascadedOpts_t opt,
                     const bool verify_decompress)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
//  printf("in_bytes %zu\n", in_bytes);
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompBatchedCascadedOpts_t options = opt;
  options.type = nvcomp::TypeOf<T>();
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

  cudaFree(d_in_data);

  if(verify_decompress == false){
    cudaFree(d_comp_out);
    return comp_out_bytes;
  }
  printf("verify_decompress\n");
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

  return comp_out_bytes;
}

template <typename data_type>
size_t _nv_compress(const uint8_t* data, const size_t size,
                    const int rle, const int delta, const bool m2_delta_mode,
                    const int bp, const int chunk_size, const bool verify_decompress){
  const data_type * cast_data = reinterpret_cast<const data_type *>(data);

  std::vector<data_type> input(cast_data, cast_data + size);
  /*
  printf("opt: %d, %d, %d, %d\n\n", rle, delta, m2_delta_mode, bp, chunk_size, verify_decompress);
  printf("input(%zu): ", input.size());
  for(auto el: input)
    printf("%u:", el);

  printf("\n");
*/
  nvcompBatchedCascadedOpts_t opt = nvcompBatchedCascadedDefaultOpts;
  opt.chunk_size = chunk_size;
  opt.num_RLEs = rle;
  opt.num_deltas = delta;
  opt.is_m2_deltas_mode = m2_delta_mode;
  opt.use_bp = bp;

  return test_cascaded<data_type>(input, opt, verify_decompress);

}

#ifdef __cplusplus
extern "C" {
#endif

size_t nv_compress(const uint8_t* data, const size_t size, const uint8_t bytes,
                   const int rle, const int delta, const bool m2_delta_mode,
                   const int bp, const int chunk_size, const bool verify_decompress){

  switch (bytes) {
    case 1:
      return _nv_compress<uint8_t>(data, size, rle, delta, m2_delta_mode, bp, chunk_size, verify_decompress);
    case 2:
      return _nv_compress<uint16_t>(data, size, rle, delta, m2_delta_mode, bp, chunk_size, verify_decompress);
    case 4:
      return _nv_compress<uint8_t>(data, size, rle, delta, m2_delta_mode, bp, chunk_size, verify_decompress);
    case 8:
      return _nv_compress<uint16_t>(data, size, rle, delta, m2_delta_mode, bp, chunk_size, verify_decompress);
  }
}

size_t test1(const int i){

  printf("get %d\n", i );
  if(i)
    return 1 << 24;
  else
    return 1 << 31;
}


void test2(const uint16_t* data, const size_t size){

  printf("size %zu\n", size );
  using data_type = uint16_t;
  std::vector<data_type> input(data, data + size);

  for(auto el: input)
    printf("%u:", el);

  printf("\n" );
}

#ifdef __cplusplus
}
#endif
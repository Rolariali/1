//
// Created by 1 on 22.04.2022.
//

#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <iostream>

// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define REQUIRE(a) assert(a)

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = (cond);                                                  \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)



template <typename T>
size_t test_cascaded(const std::vector<T>& input,
                     const nvcompBatchedCascadedOpts_t opt)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
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

static void print_options(const nvcompBatchedCascadedOpts_t & options){
  printf("chunk_size %zu, rle %d, delta %d, M2Mode %d, bp %d\n",
         options.chunk_size, options.num_RLEs, options.num_deltas, options.is_m2_deltas_mode, options.use_bp);
}
#include <random>
int main()
{
    using data_type = uint8_t;
  std::random_device rd;
  std::mt19937 random_generator(rd());
  // int8_t and uint8_t specializations of std::uniform_int_distribution are
  // non-standard, and aren't available on MSVC, so use short instead,
  // but with the range limit of the smaller type, and then cast below.
  using safe_type =
      typename std::conditional<sizeof(data_type) == 1, short, data_type>::type;
  std::uniform_int_distribution<safe_type> dist(
      0, std::numeric_limits<data_type>::max());
  std::vector<uint8_t> input = {
#include "data.h"
  };
//  for(int i=0; i < 1000*1000; i++)
//    input.push_back(static_cast<uint8_t>(dist(random_generator)));

  printf("size input %zu\n", input.size());

  // find max compressing scheme
  for(size_t chunk_size = 4096; chunk_size < 16384; chunk_size += 512)
    for(int rle = 0; rle < 5; rle++)
      for(int bp = 0; bp < 2; bp++) {
        // No delta without BitPack
        const int max_delta_num = bp == 0 ? 1 : 5;
        for (int delta = 0; delta < max_delta_num; delta++) {
          // No delta mode without delta nums
          const int max_delta_mode = delta == 0 ? 1 : 2;
          for (int delta_mode = 0; delta_mode < max_delta_mode; delta_mode++) {
            if((rle + bp + delta) == 0)
              continue;

            const nvcompBatchedCascadedOpts_t options = {chunk_size, NVCOMP_TYPE_UCHAR, rle, delta, static_cast<bool>(delta_mode), bp};

            print_options(options);
            size_t size = test_cascaded<uint8_t>(input, options);
            printf("\nsize %zu\n", size);
          }
        }
      }
    printf("\ndone\n");
}
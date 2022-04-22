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

template <typename T>
std::vector<T> data_stair( const T start, const int64_t step,
                   const T  base, const size_t min_count ){

  std::vector<T> input;
  for (int i = 0; i < min_count; i++)
    input.push_back(start + (i*step) % base);

//  printf("in(%zu): ", input.size());
//  for (auto el : input)
//    std::cout << (int)el << ":";
//
//  std::cout << std::endl;

  return input;
}

template <typename T>
void stair_delta_bp_test(const T start, const int64_t step,
                         const T  base, const size_t min_count,
                         const size_t expect_size_common_delta,
                         const size_t expect_size_m2_delta){
  size_t size_common_delta = 0;
  size_t size_m2_delta = 0;
  nvcompBatchedCascadedOpts_t opt = nvcompBatchedCascadedDefaultOpts;
  opt.num_RLEs = 0;
  opt.use_bp = 1;
  opt.num_deltas = 1;

  std::cout << start << " | " << step << " | " << base << " | " << min_count << " | "
            << expect_size_common_delta << " | " <<  expect_size_m2_delta  << std::endl;

  using data_type = T;

  auto input = data_stair<data_type>(start, step, base, min_count);
  opt.is_m2_deltas_mode = false;
  printf("@@ %zu %zu\n\n",  input.size(), min_count);
  for (auto el : input)
    std::cout << (int16_t )el << ".";


  size_common_delta = test_cascaded<data_type>(input, opt);
  printf("size_common_delta: %zu\n", size_common_delta);
  REQUIRE(expect_size_common_delta == size_common_delta);

  opt.is_m2_deltas_mode = true;
  size_m2_delta = test_cascaded<data_type>(input, opt);
  printf("size_m2_delta: %zu\n", size_m2_delta);
  REQUIRE(expect_size_m2_delta == size_m2_delta);

  REQUIRE(size_m2_delta < size_common_delta);
}
#include <limits>

template <typename T>
void test_unsigned(const char * name,
                   const size_t expect_size_common_delta,
                   const size_t expect_size_m2_delta){
  using unsignedT = std::make_unsigned_t<T>;
  using signedT = std::make_signed_t<T>;
  std::cout << "test type " << name << " / " << typeid(T).name() << std::endl;
  const T _maxU = std::numeric_limits<unsignedT>::max();
  const T _minU = std::numeric_limits<unsignedT>::min();
  const T _maxS = std::numeric_limits<signedT>::max();
  const T _minS = std::numeric_limits<signedT>::min();
  const size_t count = 101;
  const T base = 32;

  stair_delta_bp_test<T>(_minU, 1, base, count, expect_size_common_delta, expect_size_m2_delta);
  stair_delta_bp_test<T>(_maxS - base/2, 1, base, count, expect_size_common_delta, expect_size_m2_delta);
  stair_delta_bp_test<T>(_maxS - base/2, 1, base, count, expect_size_common_delta, expect_size_m2_delta);

  stair_delta_bp_test<T>(_minU + base/2, -1, base, count, expect_size_common_delta, expect_size_m2_delta);
  stair_delta_bp_test<T>(base, -1, base, count, expect_size_common_delta, expect_size_m2_delta);
  stair_delta_bp_test<T>(_minS + base/2, -1, base, count, expect_size_common_delta, expect_size_m2_delta);
}


int main()
{

  test_unsigned<uint8_t>("uint8_t", 3952, 200);
  test_unsigned<uint16_t>("uint16_t", 0, 0);
  test_unsigned<uint32_t>("uint32_t", 0, 0);
  test_unsigned<uint64_t>("uint64_t", 0, 0);

    printf("\ndone\n");
}
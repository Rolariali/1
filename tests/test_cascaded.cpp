/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define CATCH_CONFIG_MAIN

#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include "catch.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = (cond);                                                  \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
std::vector<T> buildRuns(const size_t numRuns, const size_t runSize)
{
  std::vector<T> input;
  for (size_t i = 0; i < numRuns; i++) {
    for (size_t j = 0; j < runSize; j++) {
      input.push_back(static_cast<T>(i));
    }
  }

  return input;
}

template <typename T>
void test_cascaded(const std::vector<T>& input, nvcompType_t data_type)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompBatchedCascadedOpts_t options = nvcompBatchedCascadedDefaultOpts;
  options.type = data_type;
  options.num_RLEs = 0;
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
}


template <typename T>
void test_cascaded_opt(const std::vector<T>& input, const nvcompBatchedCascadedOpts_t options )
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

//  nvcompBatchedCascadedOpts_t options = nvcompBatchedCascadedDefaultOpts;
//  options.type = data_type;
//  options.num_RLEs = 0;
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
  cudaStreamDestroy(stream);
}

} // namespace

/******************************************************************************
 * UNIT TESTS *****************************************************************
 *****************************************************************************/
/*
TEST_CASE("comp/decomp cascaded-small", "[nvcomp]")
{
  using T = int;

  std::vector<T> input = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 2, 3, 3};

  test_cascaded(input, NVCOMP_TYPE_INT);
}

TEST_CASE("comp/decomp cascaded-1", "[nvcomp]")
{
  using T = int;

  const int num_elems = 500;
  std::vector<T> input;
  for (int i = 0; i < num_elems; ++i) {
    input.push_back(i >> 2);
  }

  test_cascaded(input, NVCOMP_TYPE_INT);
}

TEST_CASE("comp/decomp cascaded-all-small-sizes", "[nvcomp][small]")
{
  using T = uint8_t;

  for (int total = 1; total < 4096; ++total) {
    std::vector<T> input = buildRuns<T>(total, 1);
    test_cascaded(input, NVCOMP_TYPE_UCHAR);
  }
}

TEST_CASE("comp/decomp cascaded-multichunk", "[nvcomp][large]")
{
  using T = int;

  for (int total = 10; total < (1 << 24); total = total * 2 + 7) {
    std::vector<T> input = buildRuns<T>(total, 10);
    test_cascaded(input, NVCOMP_TYPE_INT);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint8", "[nvcomp][small]")
{
  using T = uint8_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, NVCOMP_TYPE_UCHAR);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint16", "[nvcomp][small]")
{
  using T = uint16_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, NVCOMP_TYPE_USHORT);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint32", "[nvcomp][small]")
{
  using T = uint32_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, NVCOMP_TYPE_UINT);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint64", "[nvcomp][small]")
{
  using T = uint64_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, NVCOMP_TYPE_ULONGLONG);
  }
}

TEST_CASE("comp/decomp cascaded-none-aligned-sizes", "[nvcomp][small]")
{
  std::vector<size_t> input_sizes = { 1, 33, 1021 };

  std::vector<nvcompType_t> data_types = {
    NVCOMP_TYPE_CHAR,
    NVCOMP_TYPE_SHORT,
    NVCOMP_TYPE_INT,
    NVCOMP_TYPE_LONGLONG,
  };
  for (auto size : input_sizes) {
    std::vector<uint8_t> input = buildRuns<uint8_t>(1, size);
    for (auto type : data_types ) {
      test_cascaded(input, type);
    }
  }
}



#include <thread>

TEST_CASE("comp/decomp memory use stat", "[nvcomp][small]memory")
//  std::vector<size_t> input_sizes; //= { 1, 33, 1021 };
{
  std::vector<nvcompType_t> data_types = {
      NVCOMP_TYPE_CHAR,
      NVCOMP_TYPE_SHORT,
      NVCOMP_TYPE_INT,
      NVCOMP_TYPE_LONGLONG,
  };
  int i = 0;
  for (size_t input_sizes=0; input_sizes < 50000; input_sizes +=60) {
    input_sizes %= 49000;
    printf("input_sizes %zu\n", input_sizes);
    std::vector<uint8_t> input = buildRunsPsedoRandom<uint8_t>(1, input_sizes);
    for (auto type : data_types ) {
      test_cascaded(input, type);
    }
    printf("sleep 2s, ");
    std::this_thread::sleep_for(2000ms);
    i++;
    if(i == 100){
      i = 0;
      printf("sleep 20s");
      std::this_thread::sleep_for(20000ms);
    }

  }
}

TEST_CASE("simple memory", "simple memory")
{
  return;
  using T = int64_t;
  std::vector<T> input(0xFFFF, 3);
  while(1) {
    // create GPU only input buffer
    T* d_in_data;
    const size_t in_bytes = sizeof(T) * input.size();
    CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));
    std::this_thread::sleep_for(1000ms);
    T* copied = 0;
    CUDA_CHECK(cudaMalloc(&copied, in_bytes));
    CUDA_CHECK(
        cudaMemcpy(copied, d_in_data, in_bytes, cudaMemcpyDeviceToDevice));
    std::this_thread::sleep_for(1000ms);
    cudaFree(d_in_data);
    std::this_thread::sleep_for(1000ms);
    cudaFree(copied);
    std::this_thread::sleep_for(1000ms);
    printf("@..");
  }
}
*/

static void print_options(const nvcompBatchedCascadedOpts_t & options){
  printf("chunk_size %zu, rle %d, delta %d, M2Mode %d, bp %d\n",
         options.chunk_size, options.num_RLEs, options.num_deltas, 0, options.use_bp);
}

template <typename T>
std::vector<T> buildRunsPsedoRandom(const size_t numRuns, const size_t runSize)
{
  std::vector<T> input;
  T val;
  for (size_t i = 1; i < numRuns; i++) {
    for (size_t j = 1; j < runSize; j++) {
      val = val + i + j;
      input.push_back(static_cast<T>(val));
    }
  }

  return input;
}

TEST_CASE("comp/decomp cascade find max", "cascade loop of max")
{
  std::vector<uint8_t> input = buildRunsPsedoRandom<uint8_t>(1000, 1000);
  printf("input size: %zu\n", input.size());

  for(size_t chunk_size = 512; chunk_size < 16384; chunk_size += 512)
    for(int rle = 0; rle < 3; rle++)
      for(int bp = 0; bp < 2; bp++) {
        // No delta without BitPack
        const int max_delta_num = bp == 0 ? 1 : 5;
        for (int delta = 0; delta < max_delta_num; delta++) {
          // No delta mode without delta nums
          const int max_delta_mode = delta == 0 ? 1 : 2;
          for (int delta_mode = 0; delta_mode < max_delta_mode; delta_mode++) {
            if ((rle + bp + delta) == 0)
              continue;
            const nvcompBatchedCascadedOpts_t options = {chunk_size, NVCOMP_TYPE_UCHAR, rle, delta, bp};
            print_options(options);
            test_cascaded_opt(input, options);
          }
        }
      }

}

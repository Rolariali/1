//
// Created by 1 on 09.03.2022.
//
/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "../src/common.h"
//#include "catch.hpp"
#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"
#include <cuda_runtime.h>

#include <cstdint>
#include <random>
#include <vector>

using run_type = uint16_t;
using nvcomp::roundUpToAlignment;

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return;                                                                \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

template <typename data_type>
std::vector<data_type> generate_predefined_input_host(
    std::vector<data_type> values, std::vector<size_t> repititions)
{
  std::vector<data_type> generated_input;
  for (size_t val_idx = 0; val_idx < values.size(); val_idx++) {
    // Add `repititions[val_idx]` copies of `values[val_idx]`
    for (size_t repitition_idx = 0; repitition_idx < repititions[val_idx];
         repitition_idx++) {
      generated_input.push_back(values[val_idx]);
    }
  }
  return generated_input;
}

size_t max_compressed_size(size_t uncompressed_size)
{
  return (uncompressed_size + 3) / 4 * 4 + 4;
}

/*
 * This test case tests the correctness of batched cascaded compressor and
 * decompressor on predefined data. The test case uses 2 RLE layers, 1 Delta
 * layer, and optionally bitpacking depending on the `use_bp` argument. It first
 * compresses the data and verifies the compressed buffers. Then it decompresses
 * the data and compares against the original values.
 */
template <typename data_type>
void test_predefined_cases()
{
  // Generate input data and copy it to device memory

  std::vector<data_type> input0_host = generate_predefined_input_host(
      std::vector<data_type>{3, 9, 4, 0, 1},
      std::vector<size_t>{1, 20, 13, 25, 6});

  std::vector<data_type> input1_host = generate_predefined_input_host(
      std::vector<data_type>{1, 2, 3, 4, 5, 6},
      std::vector<size_t>{10, 6, 15, 1, 13, 9});

  void* input0_device;
  CUDA_CHECK(
      cudaMalloc(&input0_device, input0_host.size() * sizeof(data_type)));
  CUDA_CHECK(cudaMemcpy(
      input0_device,
      input0_host.data(),
      input0_host.size() * sizeof(data_type),
      cudaMemcpyHostToDevice));

  void* input1_device;
  CUDA_CHECK(
      cudaMalloc(&input1_device, input1_host.size() * sizeof(data_type)));
  CUDA_CHECK(cudaMemcpy(
      input1_device,
      input1_host.data(),
      input1_host.size() * sizeof(data_type),
      cudaMemcpyHostToDevice));

  printf("input0_host: ");
  for(auto el: input0_host)
    printf("%d:", el);

  printf("\n");

  printf("input1_host: ");
  for(auto el: input1_host)
    printf("%d:", el);

  printf("\n");

  // Copy uncompressed pointers and sizes to device memory

  std::vector<void*> uncompressed_ptrs_host
      = {input0_device, input1_device, input0_device};
  std::vector<size_t> uncompressed_bytes_host
      = {input0_host.size() * sizeof(data_type),
         input1_host.size() * sizeof(data_type),
         input0_host.size() * sizeof(data_type)};
  const size_t batch_size = uncompressed_ptrs_host.size();

  void** uncompressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&uncompressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_ptrs_device,
      uncompressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* uncompressed_bytes_device;
  CUDA_CHECK(
      cudaMalloc(&uncompressed_bytes_device, sizeof(size_t) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_bytes_device,
      uncompressed_bytes_host.data(),
      sizeof(size_t) * batch_size,
      cudaMemcpyHostToDevice));

  // Allocate compressed buffers and sizes

  std::vector<void*> compressed_ptrs_host;
  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    void* compressed_ptr;
    CUDA_CHECK(cudaMalloc(
        &compressed_ptr,
        max_compressed_size(uncompressed_bytes_host[partition_idx])*4));
    compressed_ptrs_host.push_back(compressed_ptr);
  }

  void** compressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&compressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      compressed_ptrs_device,
      compressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* compressed_bytes_device;
  CUDA_CHECK(cudaMalloc(&compressed_bytes_device, sizeof(size_t) * batch_size));

  // Launch batched compression

  nvcompBatchedCascadedOpts_t comp_opts
      = {batch_size, nvcomp::TypeOf<data_type>(), 0, 1, 0};

  auto status = nvcompBatchedCascadedCompressAsync(
      uncompressed_ptrs_device,
      uncompressed_bytes_device,
      0, // not used
      batch_size,
      nullptr, // not used
      0,       // not used
      compressed_ptrs_device,
      compressed_bytes_device,
      comp_opts,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  // Verify compressed bytes alignment

  std::vector<size_t> compressed_bytes_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
      compressed_bytes_host.data(),
      compressed_bytes_device,
      sizeof(size_t) * batch_size,
      cudaMemcpyDeviceToHost));

  for (auto const& compressed_bytes_partition : compressed_bytes_host) {
    REQUIRE(compressed_bytes_partition % 4 == 0);
    REQUIRE(compressed_bytes_partition % sizeof(data_type) == 0);
  }

  // Check the test case is small enough to fit inside one batch
  constexpr size_t chunk_size = 4096;
  for (auto const& uncompressed_byte : uncompressed_bytes_host) {
    REQUIRE(uncompressed_byte <= chunk_size);
  }

  for(int i=0; i < batch_size; i++) {
    size_t _size = compressed_bytes_host[i];
    printf("%d compressed_data_host %zu: ", i, _size);

    std::vector<data_type> compressed_data_host(_size);
    CUDA_CHECK(cudaMemcpy(
        compressed_data_host.data(),
        compressed_ptrs_host[i],
        _size - 0,
        cudaMemcpyDeviceToHost));

    for (auto el : compressed_data_host) {
      printf("%d:", el);
    }
    printf("\n");
  }
    // Cleanup

  CUDA_CHECK(cudaFree(input0_device));
  CUDA_CHECK(cudaFree(input1_device));
  CUDA_CHECK(cudaFree(uncompressed_ptrs_device));
  CUDA_CHECK(cudaFree(uncompressed_bytes_device));
  for (void* const& ptr : compressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(compressed_ptrs_device));
  CUDA_CHECK(cudaFree(compressed_bytes_device));

}

int main()
{
  test_predefined_cases<int8_t>();
}
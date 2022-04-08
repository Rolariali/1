#include "../src/common.h"
#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>
#include <iostream>

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
    printf("\nverify:\n");
    for (size_t element_idx = 0; element_idx < num_elements; element_idx++) {
//      printf("%u\t%d == %d\n", element_idx,
//             decompressed_data_host[element_idx],
//                              uncompressed_data_host[partition_idx][element_idx]);
      REQUIRE(
          decompressed_data_host[element_idx]
          == uncompressed_data_host[partition_idx][element_idx]);
    }
  }
}

template <typename data_type>
size_t test_predefined_cases(std::vector<data_type> input0_host,int rle, int delta, int bp)
{

  void* input0_device;
  CUDA_CHECK(
      cudaMalloc(&input0_device, input0_host.size() * sizeof(data_type)));
  CUDA_CHECK(cudaMemcpy(
      input0_device,
      input0_host.data(),
      input0_host.size() * sizeof(data_type),
      cudaMemcpyHostToDevice));

  // Copy uncompressed pointers and sizes to device memory

  std::vector<void*> uncompressed_ptrs_host
      = {input0_device};
  std::vector<size_t> uncompressed_bytes_host
      = {input0_host.size() * sizeof(data_type)
      };
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
      = {batch_size, nvcomp::TypeOf<data_type>(), rle, delta, bp};

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
    printf("output compressed data(size:%zu): ", _size);

    std::vector<uint8_t> compressed_data_host(_size);
    CUDA_CHECK(cudaMemcpy(
        compressed_data_host.data(),
        compressed_ptrs_host[i],
        _size - 0,
        cudaMemcpyDeviceToHost));

    for (auto el : compressed_data_host) {
      printf("%u:", el);
    }
    printf("\n");
  }

  // Check uncompressed bytes stored in the compressed buffer

  size_t* decompressed_bytes_device;
  CUDA_CHECK(
      cudaMalloc(&decompressed_bytes_device, sizeof(size_t) * batch_size));

  status = nvcompBatchedCascadedGetDecompressSizeAsync(
      compressed_ptrs_device,
      compressed_bytes_device,
      decompressed_bytes_device,
      batch_size,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  verify_decompressed_sizes(
      batch_size, decompressed_bytes_device, uncompressed_bytes_host);

  // Allocate decompressed buffers

  std::vector<void*> decompressed_ptrs_host;
  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    void* decompressed_ptr;
    CUDA_CHECK(
        cudaMalloc(&decompressed_ptr, uncompressed_bytes_host[partition_idx]));
    decompressed_ptrs_host.push_back(decompressed_ptr);
  }

  void** decompressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&decompressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      decompressed_ptrs_device,
      decompressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMemset(decompressed_bytes_device, 0, sizeof(size_t) * batch_size));

  // Launch decompression

  nvcompStatus_t* compression_statuses_device;
  CUDA_CHECK(cudaMalloc(
      &compression_statuses_device, sizeof(nvcompStatus_t) * batch_size));

  status = nvcompBatchedCascadedDecompressAsync(
      compressed_ptrs_device,
      compressed_bytes_device,
      uncompressed_bytes_device,
      decompressed_bytes_device,
      batch_size,
      nullptr, // not used
      0,       // not used
      decompressed_ptrs_device,
      compression_statuses_device,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  std::vector<nvcompStatus_t> compression_statuses_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
      compression_statuses_host.data(),
      compression_statuses_device,
      sizeof(nvcompStatus_t) * batch_size,
      cudaMemcpyDeviceToHost));

  for (auto const& compression_status : compression_statuses_host)
    REQUIRE(compression_status == nvcompSuccess);

  // Verify decompression outputs match the original uncompressed data

  std::vector<const data_type*> uncompressed_data_host
      = {input0_host.data()};

  verify_decompressed_sizes(
      batch_size, decompressed_bytes_device, uncompressed_bytes_host);

  verify_decompressed_output(
      batch_size,
      decompressed_ptrs_host,
      uncompressed_data_host,
      uncompressed_bytes_host);

  // Cleanup
  printf("cleanup\n\n");

  CUDA_CHECK(cudaFree(input0_device));
  CUDA_CHECK(cudaFree(uncompressed_ptrs_device));
  CUDA_CHECK(cudaFree(uncompressed_bytes_device));
  for (void* const& ptr : compressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(compressed_ptrs_device));
  CUDA_CHECK(cudaFree(compressed_bytes_device));
  for (void* const& ptr : decompressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(decompressed_bytes_device));
  CUDA_CHECK(cudaFree(decompressed_ptrs_device));
  CUDA_CHECK(cudaFree(compression_statuses_device));

  return compressed_bytes_host[0];
}

template <typename data_type>
void test_stair_case(const data_type start, const int step,
                     const data_type base, const size_t count, const char* name){
  size_t size;
  int rle = 0;  int delta = 1;   int bp = 0;

  std::vector<data_type> input;
  for (int i = 0; i < count; i++)
    input.push_back(start + (i*step) % base);

  printf("\n====================================================\n");
  printf("delta for %s:\n", name);
  printf("input data(size:%zu) : ", input.size());
  for (auto el : input)
    std::cout << el << ":";
//    printf("%u:", el);
  printf("\n");
  size = test_predefined_cases<data_type>(input, rle, delta, bp);
  printf("result compressed size: %zu\n", size);
  printf("\n====================================================\n");
}


int main()
{
  test_stair_case<uint8_t>(0, 20, 120, 20, "u8 20");
}

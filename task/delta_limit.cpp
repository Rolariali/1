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

bool one_only = false;

template <typename data_type>
void test_stair_case(const data_type start, const int step,
                     const data_type base, const size_t min_count, const char* name){
  size_t size;
  int rle = 0;  int delta = 1;   int bp = 0;

  std::vector<data_type> input;
  for (int i = 0; i < min_count; i++)
    input.push_back(start + (i*step) % base);

  for (int i = min_count; i < 4040; i++) {
    input.push_back(start + (i*step) % base);
    printf("\n====================================================\n");
    printf("delta for %s: stair test \n", name);
    printf("input data(size:%zu) : ", input.size());
    for (auto el : input)
      std::cout << (int)el << ":";
    //    printf("%u:", el);
    printf("\n");
    size = test_predefined_cases<data_type>(input, rle, delta, bp);
    printf("result compressed size: %zu\n", size);
    printf("\n====================================================\n");

    if(one_only)
      break;
  }
}

void test_u8(){
  test_stair_case<uint8_t>(0, 20, 120, 20, "u8 0, 20, 120, 20");
  test_stair_case<uint8_t>(200, 20, 120, 20, "u8 200, 20, 120, 20");
  test_stair_case<uint8_t>(100, 20, 119, 20, "u8 100, 20, 119, 20");
  test_stair_case<uint8_t>(220, 20, 119, 20, "u8 220, 20, 119, 20");

  test_stair_case<uint8_t>(1, 1, 254, 20, "u8 1, 1, 254, 20");
  test_stair_case<uint8_t>(0, 1, 254, 20, "u8 0, 1, 254, 20");

  test_stair_case<uint8_t>(250, 1, 127+7, 20, "u8 250, 1, 127+7, 20");
  test_stair_case<uint8_t>(255, 1, 127+2, 20, "u8 250, 1, 127+3, 20");

  test_stair_case<uint8_t>(128, 1, 255, 20, "u8 128, 1, 255, 20");

  test_stair_case<uint8_t>(127, 1, 129, 20, "u8 127, 1, 129, 20");



  test_stair_case<uint8_t>(127, -1, 255, 20, "u8 127, -1, 255, 2220");
  test_stair_case<uint8_t>(128, -1, 128+1, 20, "u8 128, -1, 128+1, 2220");

  test_stair_case<uint8_t>(0, -1, 128+1, 20, "u8 0, -1, 128+1, 2220");

  test_stair_case<uint8_t>(255, -1, 255, 20, "u8 255, -1, 255, 2220");


  test_stair_case<uint8_t>(254, -2, 254, 20, "u8 254, -2, 254, 222");
  test_stair_case<uint8_t>(220, -20, 120, 20, "u8 220, -20, 120, 222");
  test_stair_case<uint8_t>(220, -20, 119, 20, "u8 220, -20, 119, 2001");
  test_stair_case<uint8_t>(60, -20, 119, 20, "u8 60, -20, 119, 2001");
}

void test_i8(){
  test_stair_case<int8_t>(0, 1, 254, 20, "i8 0, 1, 254, 20");
  test_stair_case<int8_t>(127, 1, 254, 20, "i8 127, 1, 254, 20");
  test_stair_case<int8_t>(-128, 1, 254, 20, "i8 -128, 1, 254, 20");
  test_stair_case<int8_t>(-127, 1, 254, 20, "i8");

  test_stair_case<int8_t>(0, -1, 254, 20, "i8 0, -1, 254, 20");
  test_stair_case<int8_t>(127, -1, 254, 20, "i8 127, -1, 254, 20");
  test_stair_case<int8_t>(-128, -1, 254, 20, "i8 -128, -1, 254, 20");
  test_stair_case<int8_t>(-127, -1, 254, 20, "i8 -127, -1, 254, 20");

  test_stair_case<int8_t>(0, +20, 201, 20, "i8 0, +20, 201, 20");
  test_stair_case<int8_t>(127, 3, 254, 20, "i8 127, 3, 254, 20");
  test_stair_case<int8_t>(-128, 6, 214, 20, "i8 -128, 6, 214, 20");
  test_stair_case<int8_t>(-127, 11, 111, 20, "i8 -127, 11, 111, 20");

  test_stair_case<int8_t>(0, -4, 25, 20, "i8 0, -4, 25, 20");
  test_stair_case<int8_t>(127, -5, 21, 20, "i8 127, -5, 21, 20");
  test_stair_case<int8_t>(-128, -3, 24, 20, "i8 -128, -3, 24, 20");
  test_stair_case<int8_t>(-127, -12, 24, 20, "i8 -127, -12, 24, 20");
}

void test_u16(){

  test_stair_case<uint16_t>(0, 200, 0xFFFF - 1, 2000, "u16");
  test_stair_case<uint16_t>(0, 211, 0xFFFF - 1, 2000, "u16");
  test_stair_case<uint16_t>(0x7FFF, 444, 0xFFFF - 1, 2000, "u16");
  test_stair_case<uint16_t>(0x7FFF + 1, 900, 0xFFFF - 1, 2000, "u16");
  test_stair_case<uint16_t>(0x8FFF, 333, 0xFFFF - 1, 2000, "u16");
  test_stair_case<uint16_t>(0xFFFF, 333, 0xFFFF - 1, 2000, "u16");
  test_stair_case<uint16_t>(0, 200, 0xFFF - 1, 2000, "u16");
  test_stair_case<uint16_t>(0, 211, 0xF00F - 1, 2000, "u16");


  test_stair_case<uint16_t>(0xFFFF, -333, 0xFFFF - 1, 2000, "u16");
  test_stair_case<uint16_t>(0xFFFF, -333, 0xF2F - 1, 2000, "u16");
  test_stair_case<uint16_t>(0xFFFF, -1, 0xFF, 2000, "u16");
  test_stair_case<uint16_t>(0x7FFF, -1, 0xFF, 2000, "u16");
  test_stair_case<uint16_t>(0x7FFF + 1, -1, 0xFF, 2000, "u16");
  test_stair_case<uint16_t>(0x7FFF + 1, -1, 0xFF, 2000, "u16");
  test_stair_case<uint16_t>(0, -1, 0xFFF, 2000, "u16");
  test_stair_case<uint16_t>(0, -111, 0xFF, 2000, "u16");
}


void test_i16(){

  test_stair_case<int16_t>(0, 200, 0xFFFF - 1, 2000, "u16");
  test_stair_case<int16_t>(0, 211, 0xFFFF - 1, 2000, "u16");
  test_stair_case<int16_t>(0x7FFF, 444, 0xFFFF - 1, 2000, "u16");
  test_stair_case<int16_t>(0x7FFF + 1, 900, 0xFFFF - 1, 2000, "u16");
  test_stair_case<int16_t>(0x8FFF, 333, 0xFFFF - 1, 2000, "u16");
  test_stair_case<int16_t>(0xFFFF, 333, 0xFFFF - 1, 2000, "u16");
  test_stair_case<int16_t>(0, 200, 0xFFF - 1, 2000, "u16");
  test_stair_case<int16_t>(0, 211, 0xF00F - 1, 2000, "u16");


  test_stair_case<int16_t>(0xFFFF, -333, 0xFFFF - 1, 2000, "u16");
  test_stair_case<int16_t>(0xFFFF, -333, 0xF2F - 1, 2000, "u16");
  test_stair_case<int16_t>(0xFFFF, -1, 0xFF, 2000, "u16");
  test_stair_case<int16_t>(0x7FFF, -1, 0xFF, 2000, "u16");
  test_stair_case<int16_t>(0x7FFF + 1, -1, 0xFF, 2000, "u16");
  test_stair_case<int16_t>(0x7FFF + 1, -1, 0xFF, 2000, "u16");
  test_stair_case<int16_t>(0, -1, 0xFFF, 2000, "u16");
  test_stair_case<int16_t>(0, -111, 0xFF, 2000, "u16");
}

void test_i32(){

  test_stair_case<int32_t>(0, 2000, 0xFFFFFFFF - 1, 2000, "i32");
  test_stair_case<int32_t>(0, 211, 0xFFFFFFFF - 1, 2000, "i32");
  test_stair_case<int32_t>(0x7FFF, 444, 0xFFFFFFFF - 1, 2000, "i32");
  test_stair_case<int32_t>(0x7FFF + 1, 900, 0xFFFFFFFF - 1, 2000, "i32");
  test_stair_case<int32_t>(0x8FFF, 3330, 0xFFFFFF - 1, 2000, "i32");
  test_stair_case<int32_t>(0xFFFF, 3330, 0xFFFFF - 1, 2000, "i32");
  test_stair_case<int32_t>(0, 200, 0xFFFFF - 1, 2000, "i32");
  test_stair_case<int32_t>(0, 211, 0xFFFF - 1, 2000, "i32");


  test_stair_case<int32_t>(0xFFFF, -333, 0xFFFF - 1, 2000, "i32");
  test_stair_case<int32_t>(0xFFFF, -333, 0xF2F - 1, 2000, "i32");
  test_stair_case<int32_t>(0xFFFF, -1, 0xFF, 2000, "i32");
  test_stair_case<int32_t>(0x7FFF, -1, 0xFF, 2000, "i32");
  test_stair_case<int32_t>(0x7FFF + 1, -1, 0xFF, 2000, "i32");
  test_stair_case<int32_t>(0x7FFF + 1, -1, 0xFF, 2000, "i32");
  test_stair_case<int32_t>(0, -1, 0xFFF, 2000, "i32");
  test_stair_case<int32_t>(0, -111, 0xFF, 2000, "i32");
}



void test_u32(){

  test_stair_case<uint32_t>(0, 2000, 0xFFFFFFFF - 1, 2000, "u32");
  test_stair_case<uint32_t>(0, 211, 0xFFFFFFFF - 1, 2000, "u32");
  test_stair_case<uint32_t>(0x7FFF, 444, 0xFFFFFFFF - 1, 2000, "u32");
  test_stair_case<uint32_t>(0x7FFF + 1, 900, 0xFFFFFFFF - 1, 2000, "u32");
  test_stair_case<uint32_t>(0x8FFF, 3330, 0xFFFFFF - 1, 2000, "u32");
  test_stair_case<uint32_t>(0xFFFF, 3330, 0xFFFFF - 1, 2000, "u32");
  test_stair_case<uint32_t>(0, 200, 0xFFFFF - 1, 2000, "u32");
  test_stair_case<uint32_t>(0, 211, 0xFFFF - 1, 2000, "u32");


  test_stair_case<uint32_t>(0xFFFF, -333, 0xFFFF - 1, 2000, "u32");
  test_stair_case<uint32_t>(0xFFFF, -333, 0xF2F - 1, 2000, "u32");
  test_stair_case<uint32_t>(0xFFFF, -1, 0xFF, 2000, "u32");
  test_stair_case<uint32_t>(0x7FFF, -1, 0xFF, 2000, "u32");
  test_stair_case<uint32_t>(0x7FFF + 1, -1, 0xFF, 2000, "u32");
  test_stair_case<uint32_t>(0x7FFF + 1, -1, 0xFF, 2000, "u32");
  test_stair_case<uint32_t>(0, -1, 0xFFF, 2000, "u32");
  test_stair_case<uint32_t>(0, -111, 0xFF, 2000, "u32");
}

int main()
{
  one_only = true;

  test_i8();
  test_u8();


  test_u16();
  test_i16();

  test_i32();
  test_u32();

  printf("\ndone\n");
}

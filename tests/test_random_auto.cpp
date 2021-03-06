/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp.h"
#include "nvcomp/cascaded.h"
#include "nvcomp/cascaded.hpp"

#include "test_common.h"

#include <cuda_profiler_api.h>
#include <iomanip>
#include <random>
#include <thread>
#include <vector>

// Test method that takes an input data, compresses it (on the CPU),
// decompresses it on the GPU, and verifies it is correct.
// Uses C API Cascaded Compression with automatic format selector
template <typename T>
void test_auto_c(const std::vector<T>& data)
{
  const nvcompType_t type = nvcomp::TypeOf<T>();

#if VERBOSE > 1
  // dump input data
  std::cout << "Input" << std::endl;
  for (size_t i = 0; i < data.size(); i++)
    std::cout << data[i] << " ";
  std::cout << std::endl;
#endif

  // these two items will be the only forms of communication between
  // compression and decompression
  void* d_comp_out;
  size_t comp_out_bytes;

  {
    // this block handles compression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression
    std::cout << "----------" << std::endl;
    std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

    // create GPU only input buffer
    void* d_in_data;
    const size_t in_bytes = sizeof(T) * data.size();
    CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvcompStatus_t status;

    // Compress on the GPU
    size_t comp_temp_bytes;
    size_t metadata_bytes;

    status = nvcompCascadedCompressConfigure(
        NULL, // Null means to auto-select the best scheme
        nvcomp::TypeOf<T>(),
        in_bytes,
        &metadata_bytes,
        &comp_temp_bytes,
        &comp_out_bytes);

    void* d_comp_temp;
    CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
    CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

    status = nvcompCascadedCompressAsync(
        NULL,
        nvcomp::TypeOf<T>(),
        d_in_data,
        in_bytes,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out,
        &comp_out_bytes,
        stream);
    REQUIRE(status == nvcompSuccess);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_comp_temp);
    cudaFree(d_in_data);
    cudaStreamDestroy(stream);

    std::cout << "comp_size: " << comp_out_bytes
              << ", compressed ratio: " << std::fixed << std::setprecision(2)
              << (double)in_bytes / comp_out_bytes << std::endl;
  }

  {
    // this block handles decompression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // get metadata from compressed data
    void* metadata = NULL;
    size_t metadata_bytes;
    size_t decomp_temp_bytes;
    size_t decomp_out_bytes;

    nvcompStatus_t err = nvcompCascadedDecompressConfigure(
        d_comp_out,
        comp_out_bytes,
        &metadata,
        &metadata_bytes,
        &decomp_temp_bytes,
        &decomp_out_bytes,
        stream);
    REQUIRE(err == nvcompSuccess);

    // allocate temp buffer
    void* d_decomp_temp;
    CUDA_CHECK(cudaMalloc(
        &d_decomp_temp, decomp_temp_bytes)); // also can use RMM_ALLOC instead

    // allocate output buffer
    void* decomp_out_ptr;
    CUDA_CHECK(cudaMalloc(
        &decomp_out_ptr, decomp_out_bytes)); // also can use RMM_ALLOC instead

    auto start = std::chrono::steady_clock::now();

    // execute decompression (asynchronous)
    err = nvcompCascadedDecompressAsync(
        d_comp_out,
        comp_out_bytes,
        metadata,
        metadata_bytes,
        d_decomp_temp,
        decomp_temp_bytes,
        decomp_out_ptr,
        decomp_out_bytes,
        stream);
    REQUIRE(err == nvcompSuccess);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // stop timing and the profiler
    auto end = std::chrono::steady_clock::now();
    std::cout << "throughput (GB/s): " << gbs(start, end, decomp_out_bytes)
              << std::endl;

    nvcompCascadedDestroyMetadata(metadata);

    cudaStreamDestroy(stream);
    cudaFree(d_decomp_temp);
    cudaFree(d_comp_out);

    std::vector<T> res(decomp_out_bytes / sizeof(T));
    cudaMemcpy(
        &res[0], decomp_out_ptr, decomp_out_bytes, cudaMemcpyDeviceToHost);

#if VERBOSE > 1
    // dump output data
    std::cout << "Output" << std::endl;
    for (size_t i = 0; i < data.size(); i++)
      std::cout << ((T*)out_ptr)[i] << " ";
    std::cout << std::endl;
#endif

    REQUIRE(res == data);
  }

}


// Test method that takes an input data, compresses it (on the CPU),
// decompresses it on the GPU, and verifies it is correct.
// Uses C++ API Cascaded Compression with automatic format selector
template <typename T>
void test_auto_cpp(const std::vector<T>& data)
{
  const nvcompType_t type = nvcomp::TypeOf<T>();

#if VERBOSE > 1
  // dump input data
  std::cout << "Input" << std::endl;
  for (size_t i = 0; i < data.size(); i++)
    std::cout << data[i] << " ";
  std::cout << std::endl;
#endif

  // these two items will be the only forms of communication between
  // compression and decompression
  void* d_comp_out;
  size_t comp_out_bytes;

  {
    // this block handles compression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression
    std::cout << "----------" << std::endl;
    std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

    // create GPU only input buffer
    void* d_in_data;
    const size_t in_bytes = sizeof(T) * data.size();

    CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CascadedCompressor compressor(nvcomp::TypeOf<T>());

    size_t comp_temp_bytes;
    compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);

    // Allocate temp storage
    void* d_comp_temp;
    CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

    // Allocate output space
    CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

    compressor.compress_async(
        d_in_data,
        in_bytes,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out,
        &comp_out_bytes,
        stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_comp_temp);
    cudaFree(d_in_data);
    cudaStreamDestroy(stream);

    std::cout << "comp_size: " << comp_out_bytes
              << ", compressed ratio: " << std::fixed << std::setprecision(2)
              << (double)in_bytes / comp_out_bytes << std::endl;
  }

  {
    // this block handles decompression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t decomp_temp_bytes;
    size_t decomp_out_bytes;
    CascadedDecompressor decompressor;

    decompressor.configure(
        d_comp_out,
        comp_out_bytes,
        &decomp_temp_bytes,
        &decomp_out_bytes,
        stream);
    REQUIRE(decomp_temp_bytes > 0);
    REQUIRE(decomp_out_bytes > 0);


    // allocate temp buffer
    void* d_decomp_temp;
    CUDA_CHECK(cudaMalloc(
        &d_decomp_temp, decomp_temp_bytes)); // also can use RMM_ALLOC instead

    // allocate output buffer
    T* decomp_out_ptr;
    CUDA_CHECK(cudaMalloc(
        &decomp_out_ptr, decomp_out_bytes)); // also can use RMM_ALLOC instead

    auto start = std::chrono::steady_clock::now();

    // execute decompression 
    decompressor.decompress_async(
        d_comp_out, comp_out_bytes, d_decomp_temp, decomp_temp_bytes, decomp_out_ptr, decomp_out_bytes, stream);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // stop timing and the profiler
    auto end = std::chrono::steady_clock::now();
    std::cout << "throughput (GB/s): " << gbs(start, end, decomp_out_bytes)
              << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(d_decomp_temp);
    cudaFree(d_comp_out);

    //  int* res = (int*)malloc(decomp_bytes);
    std::vector<T> res(decomp_out_bytes / sizeof(T));
    cudaMemcpy(
        &res[0], decomp_out_ptr, decomp_out_bytes, cudaMemcpyDeviceToHost);

#if VERBOSE > 1
    // dump output data
    std::cout << "Output" << std::endl;
    for (size_t i = 0; i < data.size(); i++)
      std::cout << ((T*)out_ptr)[i] << " ";
    std::cout << std::endl;
#endif

    REQUIRE(res == data);
  }

}



template <typename T>
void test_random_auto_c(int max_val, int max_run, size_t chunk_size)
{
  std::vector<T> data;
  int seed = (max_val ^ max_run ^ chunk_size);
  random_runs(data, (T)max_val, (T)max_run, seed);
  test_auto_c(data);
}

template <typename T>
void test_random_auto_cpp(int max_val, int max_run, size_t chunk_size)
{

  std::vector<T> data;
  int seed = (max_val ^ max_run ^ chunk_size);
  random_runs(data, (T)max_val, (T)max_run, seed);

  test_auto_cpp(data); 
}


TEST_CASE("Auto-tiny-int", "[tiny][auto]")
{
  for(int size=1; size<1024; size++) {
    test_random_auto_c<int>(10,10,size);
  }
}

TEST_CASE("Auto-small-int", "[small][auto]")
{
  test_random_auto_c<int>(10,10,10000);
  test_random_auto_cpp<int>(10,10,10000);
}
TEST_CASE("Auto-large-int", "[large][auto]")
{
  test_random_auto_c<int>(10000, 1000, 10000000);
  test_random_auto_cpp<int>(10000, 1000, 10000000);
}
TEST_CASE("Auto-small-ll", "[small-ll][auto]")
{
  test_random_auto_c<int64_t>(10,10,10000);
  test_random_auto_cpp<int64_t>(10,10,10000);
}
TEST_CASE("Auto-large-ll", "[large-ll][auto]")
{
  test_random_auto_c<int64_t>(10000, 1000, 10000000);
  test_random_auto_cpp<int64_t>(10000, 1000, 10000000);
}

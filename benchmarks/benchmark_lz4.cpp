/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include "nvcomp/lz4.hpp"

#include "benchmark_common.h"

#include <string.h>
#include <string>

using namespace nvcomp;

static void print_usage()
{
  printf("Usage: benchmark_lz4 [OPTIONS]\n");
  printf("  %-35s Binary dataset filename (required).\n", "-f, --filename");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  printf("  %-35s Data type (default 'char', options are 'char', 'short', 'int')\n", "-t, --type");
  printf(
      "  %-35s Output GPU memory allocation sizes (default off)\n",
      "-m --memory");
  exit(1);
}

// Benchmark performance from the binary data file fname
static void run_benchmark(char* fname, int verbose_memory, nvcompType_t in_type)
{
  using T = uint8_t;

  size_t input_elts = 0;
  std::vector<T> data;
  data = load_dataset_from_binary<T>(fname, &input_elts);

  // Make sure dataset fits on GPU to benchmark total compression
  size_t freeMem;
  size_t totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  if (freeMem < input_elts * sizeof(T)) {
    std::cout << "Insufficient GPU memory to perform compression." << std::endl;
    exit(1);
  }

  std::cout << "----------" << std::endl;
  std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input_elts;
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

  LZ4Compressor compressor(1 << 16, in_type);

  size_t comp_temp_bytes;
  size_t comp_out_bytes;
  compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);
  benchmark_assert(
      comp_out_bytes > 0, "Output size must be greater than zero.");

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Allocate temp workspace
  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Allocate compressed output buffer
  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  if (verbose_memory) {
    std::cout << "compression memory (input+output+temp) (B): "
              << (in_bytes + comp_out_bytes + comp_temp_bytes) << std::endl;
    std::cout << "compression temp space (B): " << comp_temp_bytes << std::endl;
    std::cout << "compression output space (B): " << comp_out_bytes
              << std::endl;
  }

  // Launch compression
  size_t* d_comp_out_bytes;
  CUDA_CHECK(
      cudaMallocHost((void**)&d_comp_out_bytes, sizeof(*d_comp_out_bytes)));

  auto start = std::chrono::steady_clock::now();
  compressor.compress_async(
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      d_comp_out_bytes,
      stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();

  comp_out_bytes = *d_comp_out_bytes;

  cudaFreeHost(d_comp_out_bytes);
  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  std::cout << "comp_size: " << comp_out_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)data.size() * sizeof(T) / comp_out_bytes << std::endl;
  std::cout << "compression throughput (GB/s): "
            << gbs(start, end, data.size() * sizeof(T)) << std::endl;

  // get metadata from compressed data on GPU
  LZ4Decompressor decompressor;

  size_t decomp_temp_bytes;
  size_t decomp_bytes;
  decompressor.configure(
      d_comp_out, comp_out_bytes, &decomp_temp_bytes, &decomp_bytes, stream);

  // allocate temp buffer
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(
      &d_decomp_temp, decomp_temp_bytes)); // also can use RMM_ALLOC instead

  // allocate output buffer
  T* decomp_out_ptr;
  CUDA_CHECK(cudaMalloc(
      (void**)&decomp_out_ptr, decomp_bytes)); // also can use RMM_ALLOC instead

  // get output size
  if (verbose_memory) {
    std::cout << "decompression memory (input+output+temp) (B): "
              << (decomp_bytes + comp_out_bytes + decomp_temp_bytes)
              << std::endl;
    std::cout << "decompression temp space (B): " << decomp_temp_bytes
              << std::endl;
  }

  start = std::chrono::steady_clock::now();

  // execute decompression (asynchronous)
  decompressor.decompress_async(
      d_comp_out,
      comp_out_bytes,
      d_decomp_temp,
      decomp_temp_bytes,
      decomp_out_ptr,
      decomp_bytes,
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  end = std::chrono::steady_clock::now();

  std::cout << "decompression throughput (GB/s): "
            << gbs(start, end, decomp_bytes) << std::endl;

  cudaStreamDestroy(stream);
  cudaFree(d_decomp_temp);
  cudaFree(d_comp_out);

  benchmark_assert(
      decomp_bytes == input_elts * sizeof(T),
      "Decompressed result incorrect size.");

  std::vector<T> res(input_elts);
  cudaMemcpy(
      res.data(),
      decomp_out_ptr,
      input_elts * sizeof(T),
      cudaMemcpyDeviceToHost);

  // check the size

#if VERBOSE > 1
  // dump output data
  std::cout << "Output" << std::endl;
  for (size_t i = 0; i < data.size(); i++)
    std::cout << ((T*)out_ptr)[i] << " ";
  std::cout << std::endl;
#endif

  benchmark_assert(res == data, "Decompressed data does not match input.");
}

int main(int argc, char* argv[])
{
  char* fname = NULL;
  int gpu_num = 0;
  int verbose_memory = 0;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  // Parse command-line arguments
  char** argv_end = argv + argc;
  argv += 1;
  while (argv != argv_end) {
    char* arg = *argv++;
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-?") == 0) {
      print_usage();
      return 1;
    }
    if (strcmp(arg, "--memory") == 0 || strcmp(arg, "-m") == 0) {
      verbose_memory = 1;
      continue;
    }

    // all arguments below require at least a second value in argv
    if (argv >= argv_end) {
      print_usage();
      return 1;
    }

    char* optarg = *argv++;
    if (strcmp(arg, "--filename") == 0 || strcmp(arg, "-f") == 0) {
      fname = optarg;
      continue;
    }
    if (strcmp(arg, "--gpu") == 0 || strcmp(arg, "-g") == 0) {
      gpu_num = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--type") == 0 || strcmp(arg, "-t") == 0) {
      if (strcmp(optarg, "char") == 0) {
        data_type = NVCOMP_TYPE_CHAR;
      } else if (strcmp(optarg, "short") == 0) {
        data_type = NVCOMP_TYPE_SHORT;
      } else if (strcmp(optarg, "int") == 0) {
        data_type = NVCOMP_TYPE_INT;
      } else {
        print_usage();
        return 1;
      }
      continue;
    }

    print_usage();
    return 1;
  }

  if (fname == NULL) {
    print_usage();
  }

  cudaSetDevice(gpu_num);

  run_benchmark(fname, verbose_memory, data_type);

  return 0;
}

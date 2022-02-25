//
// Created by 1 on 19.02.2022.
//

#ifndef NVCOMP_FCASCADE_H
#define NVCOMP_FCASCADE_H


#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

#include "../src/highlevel/CascadedMetadata.h"

using namespace std;
using namespace nvcomp;

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return nvcompErrorInternal;                                                                \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      printf(                                                                  \
          "API call failure \"" #func "\" with %d at " __FILE__ ":%d\n",       \
          (int)rt,                                                             \
          __LINE__);                                                           \
      return false;                                                                \
    }                                                                          \
  } while (0)

struct CompResult{
  vector<int8_t> output;
  vector<int8_t> meta;
  size_t out_bytes;
  size_t meta_bytes;

  //free after use
  nvcomp::highlevel::CascadedMetadata* meta_ptr;
};


struct FCascade
{
  static bool cascade(const vector<int8_t> input, CompResult & res,
               const int RLE, const int deltas, const int use_bp,
                      nvcompCascadedFormatOpts * opts = NULL)
  {

    typedef int8_t T;
    const nvcompType_t type = NVCOMP_TYPE_CHAR;

    const size_t input_size = input.size();
    nvcompCascadedFormatOpts comp_opts;

    if(opts == NULL) {
      comp_opts.num_RLEs = RLE;
      comp_opts.num_deltas = deltas;
      comp_opts.use_bp = use_bp;
    } else {
      comp_opts = *opts;
    }
    // create GPU only input buffer
    void* d_in_data;
    const size_t in_bytes = sizeof(T) * input_size;
    CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
    CUDA_CHECK(cudaMemcpy(
        d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvcompStatus_t status;

    // Compress on the GPU
    size_t comp_temp_bytes;
    size_t comp_out_bytes;
    size_t metadata_bytes;
    status = nvcompCascadedCompressConfigure(
        &comp_opts,
        type,
        in_bytes,
        &metadata_bytes,
        &comp_temp_bytes,
        &comp_out_bytes);
    REQUIRE(status == nvcompSuccess);

    void* d_comp_temp;
    void* d_comp_out;
    CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
    CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

    size_t* d_comp_out_bytes;
    CUDA_CHECK(
        cudaMalloc((void**)&d_comp_out_bytes, sizeof(*d_comp_out_bytes)));
    CUDA_CHECK(cudaMemcpy(
        d_comp_out_bytes,
        &comp_out_bytes,
        sizeof(*d_comp_out_bytes),
        cudaMemcpyHostToDevice));

    status = nvcompCascadedCompressAsync(
        &comp_opts,
        type,
        d_in_data,
        in_bytes,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out,
        d_comp_out_bytes,
        stream);
    REQUIRE(status == nvcompSuccess);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(
        &comp_out_bytes,
        d_comp_out_bytes,
        sizeof(comp_out_bytes),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_comp_out_bytes));
    CUDA_CHECK(cudaFree(d_comp_temp));
    CUDA_CHECK(cudaFree(d_in_data));


    vector<uint8_t> out(comp_out_bytes);

    CUDA_CHECK(cudaMemcpy(
        out.data(), d_comp_out, comp_out_bytes, cudaMemcpyDeviceToHost));

    // get temp and output size
    size_t temp_bytes;
    size_t output_bytes;
    void* metadata_ptr = NULL;

    status = nvcompDecompressGetMetadata(
        d_comp_out, comp_out_bytes, &metadata_ptr, stream);
    REQUIRE(status == nvcompSuccess);

    res.meta_ptr = static_cast<nvcomp::highlevel::CascadedMetadata*>(metadata_ptr);


    res.meta_bytes = res.meta_ptr->getDataOffset(0);
    for(int i = 0;  i < res.meta_bytes; i++)
      res.meta.push_back(out[i]);

    res.out_bytes = comp_out_bytes - res.meta_bytes;
    // res.output.resize(res.out_bytes);

    for(int i = res.meta_bytes;  i < comp_out_bytes; i++)
      res.output.push_back(out[i]);

    return true;
  }

  static void show_stat(const vector<int8_t> input, struct CompResult & res,
                 const bool stat = true,
                 const bool show_meta = false){
    if(show_meta) {
      printf("meta: %zu \t\t:", res.meta_bytes);
      for (auto el : res.meta)
        printf("%x:", el);
    }

    printf("\ninput: %zu\t\t:", input.size());
    for(auto el: input)
      printf("%x:", el);

    printf("\ncompress: %zu %zu\t\t:", res.out_bytes, res.output.size());
    for(auto el: res.output)
      printf("%x:", el);
    printf("\n");

    if(stat == false)
      return;

    printf("stat:\n");
    printf(
        "haveAnyOffsetsBeenSet: %u\n", res.meta_ptr->haveAnyOffsetsBeenSet());
    printf(
        "haveAllOffsetsBeenSet: %u\n", res.meta_ptr->haveAllOffsetsBeenSet());

    printf("getNumInputs: %d\n", res.meta_ptr->getNumInputs());

    int ii = res.meta_ptr->getNumInputs();
    for(int i =0; i < ii; i++){
      printf("=== i %d\n", i);
      printf("getNumElementsOf: %lu\n", res.meta_ptr->getNumElementsOf(i));
      printf("haveAllOffsetsBeenSet: %d\n", res.meta_ptr->isSaved(i));
      printf("getDataOffset: %lu\n", res.meta_ptr->getDataOffset(i));
      printf("getDataType: %d\n", res.meta_ptr->getDataType(i));

      printf("getHeader length: %llu\n", res.meta_ptr->getHeader(i).length);
      printf("getHeader minValue: %d\n", res.meta_ptr->getHeader(i).minValue.i32);
      printf("getHeader numBits: %u\n", res.meta_ptr->getHeader(i).numBits);
    }

    printf("getTempBytes: %ld\n", res.meta_ptr->getTempBytes());
  }

  static void short_stat(const vector<uint8_t> input, const struct CompResult & res,
                  size_t qty = 32)
  {
    printf("\nmeta %zu\n", res.meta_bytes);
    printf("\ninput: %zu\t\t: ", input.size());
    for(int i = 0; i < input.size() && i < qty; i++)
      printf("%x:", input[i]);


    printf("\ncompress: %zu\t\t: ", res.out_bytes);
    for(int i = 0; i < res.output.size() && i < qty; i++)
      printf("%x:", res.output[i]);
  }

};

#endif // NVCOMP_FCASCADE_H

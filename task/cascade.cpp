//
// Created by 1 on 13.02.2022.
//
#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

//#include "highlevel/BitcompMetadata.h"
//#include "highlevel/CascadedMetadata.h"
#include "../src/highlevel/CascadedMetadata.h"
//#include "highlevel/Metadata.h"
// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return 0;                                                                \
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
      return 0;                                                                \
    }                                                                          \
  } while (0)




 #include "test_data.h"

int main()
{
  nvcompCascadedFormatOpts opts;
//  typedef uint8_t T;

  printf("--------------------------------------------\n");
  typedef uint8_t T;
  const nvcompType_t type = NVCOMP_TYPE_UCHAR;
  size_t min_comp_out_bytes = 99999999999;

//  T input[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};
  const size_t input_size = input.size();
  nvcompCascadedFormatOpts comp_opts;

  for (int packing = 0; packing <= 1; ++packing)
    for (int Delta = 0; Delta <= 5; ++Delta)
      for (int RLE = 0; RLE <= 5; ++RLE) {
        if (RLE + Delta + packing == 0) {
          // don't bother if there is no compression
          continue;
        }
        printf("num_RLEs %u - num_deltas %u - use_bp %u\n", RLE, Delta, packing);
//        std::cout << "num_RLEs " << RLE;      // << endl;
//        cout << " num_deltas " << Delta; // << endl;
//        cout << " use_bp " << packing << endl;
//        cout << " -------------------------------------------------------"
//             << endl;
        comp_opts.num_RLEs = RLE;
        comp_opts.num_deltas = Delta;
        comp_opts.use_bp = packing;
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

//        printf(
//            "nvcompCascadedCompressConfigure:\n"
//            "metadata_bytes %u\n"
//            "comp_temp_bytes %u\n"
//            "comp_out_bytes %u\n",
//            metadata_bytes,
//            comp_temp_bytes,
//            comp_out_bytes);

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

        // get temp and output size
        //  size_t temp_bytes;
        //  size_t output_bytes;
        //  void* metadata_ptr = NULL;

        vector<uint8_t> out(comp_out_bytes);

        CUDA_CHECK(cudaMemcpy(
            out.data(), d_comp_out, comp_out_bytes, cudaMemcpyDeviceToHost));

        printf(
            "nvcompCascadedCompressAsync:  "
            "in_bytes %u"
            "  comp_out_bytes %u\n",
            in_bytes,
            comp_out_bytes);

        if( min_comp_out_bytes > comp_out_bytes)
          min_comp_out_bytes = comp_out_bytes;

        //  for(auto el: out)
        //    printf("%x:", el);
        //
        //  printf("\nwithout meta:\n");
        //
        //  for(int i = metadata_bytes; i < comp_out_bytes; i++)
        //    printf("%x:", out[i]);
        //  printf("\nmeta\n");

        // get temp and output size
        size_t temp_bytes;
        size_t output_bytes;
        void* metadata_ptr = NULL;

        status = nvcompDecompressGetMetadata(
            d_comp_out, comp_out_bytes, &metadata_ptr, stream);
        REQUIRE(status == nvcompSuccess);

//        for (int i = 0; i < metadata_bytes; i++)
//          printf("%x:", ((uint8_t*)metadata_ptr)[i]);
//        printf("\n\n");

        status = nvcompDecompressGetTempSize(metadata_ptr, &temp_bytes);
        REQUIRE(status == nvcompSuccess);

//        nvcomp::highlevel::CascadedMetadata* m
//            = static_cast<nvcomp::highlevel::CascadedMetadata*>(metadata_ptr);
//        printf("getNumInputs: %u\n", m->getNumInputs());
//        printf("useBitPacking: %u\n", m->useBitPacking());
//        printf("haveAnyOffsetsBeenSet: %u\n", m->haveAnyOffsetsBeenSet());
//
//        printf("haveAllOffsetsBeenSet: %u\n", m->haveAllOffsetsBeenSet());
//        printf("getNumElementsOf: %u\n", m->getNumElementsOf(0));
//
//        printf("getDataOffset: %u\n", m->getDataOffset(0));
//
//        printf("isSaved: %u\n", m->isSaved(0));
//        printf("getDataType: %u\n", m->getDataType(0));
//
//        printf("getHeader: %u\n", m->getHeader(0).length);

        //  printf("\n===get data \n");
        //  size_t gd = m->getDataOffset(0);
        //  for(int i =gd;  i < comp_out_bytes; i++)
        //    printf("%x:", out[i]);
        //
        //  printf("\n\n");
        //
        //
        //  printf("\n===get data int \n");
        //  // size_t gd = m->getDataOffset(0);
        //  for(int i =gd;  i < comp_out_bytes; i++)
        //    printf("%d:", (char)out[i]);
        //
        //  printf("\n\n");

        // allocate temp buffer
        void* temp_ptr;
        CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

        status = nvcompDecompressGetOutputSize(metadata_ptr, &output_bytes);
        REQUIRE(status == nvcompSuccess);

        // allocate output buffer
        void* out_ptr;
        CUDA_CHECK(cudaMalloc(&out_ptr, output_bytes));
      }

  printf("\"min_comp_out_bytes \" << %uz", min_comp_out_bytes);

  printf("\n\ndone\n");
}

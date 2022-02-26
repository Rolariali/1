//
// Created by 1 on 19.02.2022.
//

#include "FCascade.h"

extern bool verbose;

//#include "data/big_delta.h"

int main()
{
  verbose = true;

  vector<int8_t> input
      = {99, 1, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98,
         98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 91};
  {
    printf("--------------------Delta------------------------\n");

    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 0, 1, 0));
    FCascade::show_stat(input, res, true, true);

    printf("--------------------------------------------\n");
  }
  {
    printf("--------------------Delta OVERFLOW_DELTA_FOR_INTERVAL------------------------\n");

    struct CompResult res;
    nvcompCascadedFormatOpts opts = {0,1,
                                     0, nvcompCascadedFormatOpts::DeltaOpts::DeltaMode::NORMAL_DELTA};
    opts.delta_opts.delta_mode = nvcompCascadedFormatOpts::DeltaOpts::DeltaMode::OVERFLOW_DELTA_FOR_INTERVAL;
    REQUIRE(FCascade::cascade(input, res, 0, 1, 0, &opts));
    FCascade::show_stat(input, res, true, true);

    printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("\ndecompress:\n");

    // get temp and output size
    size_t temp_bytes;
    size_t output_bytes;
    void* metadata_ptr = NULL;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvcompStatus_t status = nvcompCascadedDecompressConfigure(
        res.d_ptr_compress_data,
        res.compress_out_bytes,
        &metadata_ptr,
        &res.metadata_bytes_orig,
        &temp_bytes,
        &output_bytes,
        stream);
    REQUIRE(status == nvcompSuccess);

    // allocate temp buffer
    void* temp_ptr;
    CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));
    // allocate output buffer
    void* out_ptr;
    CUDA_CHECK(cudaMalloc(&out_ptr, output_bytes));

    // execute decompression (asynchronous)
    status = nvcompCascadedDecompressAsync(
        res.d_ptr_compress_data,
        res.compress_out_bytes,
        metadata_ptr,
        res.metadata_bytes_orig,
        temp_ptr,
        temp_bytes,
        out_ptr,
        output_bytes,
        stream);
    REQUIRE(status == nvcompSuccess);

    CUDA_CHECK(cudaDeviceSynchronize());

    nvcompCascadedDestroyMetadata(metadata_ptr);

    // Copy result back to host
    vector<int8_t> decomp(input.size());
    cudaMemcpy(decomp.data(), out_ptr, output_bytes, cudaMemcpyDeviceToHost);

    CUDA_CHECK(cudaFree(temp_ptr));
//    CUDA_CHECK(cudaFree(d_comp_out));

    // Verify correctness
    printf("\n==== decompress data: ");
    for (size_t i = 0; i < input.size(); ++i) {
      printf("%d:", decomp[i]);
      REQUIRE(decomp[i] == input[i]);
    }

    printf("--------------------------------------------\n");
  }
}
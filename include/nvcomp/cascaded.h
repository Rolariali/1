/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVCOMP_CASCADED_H
#define NVCOMP_CASCADED_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure that stores the compression configuration
 */
typedef struct
{
  /**
   * @brief The number of Run Length Encodings to perform.
   */
  int num_RLEs;

  /**
   * @brief The number of Delta Encodings to perform.
   */
  int num_deltas;

  /**
   * @brief Whether or not to bitpack the final layers.
   */
  int use_bp;

  struct DeltaOpts{
      enum DeltaMode {
        NORMAL_DELTA = 0x00,
        OVERFLOW_DELTA_FOR_INTERVAL = 0x01,
      };
      DeltaMode delta_mode = NORMAL_DELTA;
  } delta_opts;

} nvcompCascadedFormatOpts;

/**
 * @brief Configure the Cascaded compressor and return temp and output
 * sizes needed to perform the compression.  If no format is provided (i.e.,
 * NULL), temporary and output size estimates are based on the format that would
 * require the largest allocation.
 *
 * @param format_opts The cascaded format options.  If set to NULL, temporary
 * storage sizes are allocated to enable running the CascadedSelector during
 * compression.
 * @param type The data type of the uncompressed data.
 * @param uncompressed_bytes The size of the uncompressed data on the device.
 * @param metadata_bytes The bytes needed to store the metadata (output)
 * @param temp_bytes The temporary memory required for compression (output)
 * @param compressed_bytes The estaimted size of the compressed result (output)
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompCascadedCompressConfigure(
    const nvcompCascadedFormatOpts* format_opts,
    nvcompType_t type,
    size_t uncompressed_bytes,
    size_t* metadata_bytes,
    size_t* temp_bytes,
    size_t* compressed_bytes);

/**
 * @brief Perform asynchronous compression. The pointers `compressed_ptr` and
 * `compressed_bytes` must be to preallocated memory directly accessible by the
 * GPU. If no format is provided (i.e., NULL), the CascadedSelector is also run
 * to determine the best compression format and the function synchronizes on the
 * stream.
 *
 *
 * NOTE: Currently, cascaded compression is limited to 2^31-1 bytes. To
 * compress larger data, break it up into chunks.
 *
 * @param format_opts The cascaded format options. If set to NULL, the format
 * is automatically selected using the CascadedSelector.  In this case,
 * the function runs synchronously on the CUDA stream.
 * @param type The data type of the uncompressed data.
 * @param uncompressed_ptr The uncompressed data on the device.
 * @param uncompressed_bytes The size of the uncompressed data in bytes.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param compressed_ptr The location to write compresesd data to on the device.
 * @param compressed_bytes The size of the output location on input, and the
 * size of the compressed data on output. This pointer must be preallocated and
 * directly accessible by the GPU.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompCascadedCompressAsync(
    const nvcompCascadedFormatOpts* format_opts,
    nvcompType_t type,
    const void* uncompressed_ptr,
    size_t uncompressed_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* compressed_ptr,
    size_t* compressed_bytes,
    cudaStream_t stream);

/**
 * @brief Configure the decompression and get the output and temp sizes
 * needed to perform the decompression. This function allocates host-side
 * memory, synchronizes the provided CUDA stream, and blocks CPU execution until
 * the metadata is extracted and copied from the `compressed_ptr`.
 *
 * NOTE: Currently, cascaded compression is limited to 2^31-1 bytes. To
 * compress larger data, break it up into chunks.
 *
 * @param compressed_ptr The compressed data on the device.
 * @param compressed_bytes The size of the compressed data in bytes.
 * @param metadata_ptr The pointer that is to be populated with the metadata
 * needed to perform decompression.  This function allocates host-side memory
 * and copies the metdata to it.
 * @param metadata_bytes The size of the metadata that this function allocates.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param uncompressed_bytes The required size of the output location in bytes
 * (output).
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompCascadedDecompressConfigure(
    const void* compressed_ptr,
    size_t compressed_bytes,
    void** metadata_ptr,
    size_t* metadata_bytes,
    size_t* temp_bytes,
    size_t* uncompressed_bytes,
    cudaStream_t stream);

/**
 * @brief Perform the asynchronous decompression.
 *
 * @param compressed_ptr The compressed data on the device.
 * @param compressed_bytes The size of the compressed data.
 * @param metadata_ptr The metadata (accessible by host).
 * @param metadata_bytes The size of the metadata.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace.
 * @param uncompressed_ptr The output location on the device (output).
 * @param uncompressed_bytes The size of the uncompressed data as returned by
 * `nvcompLZ4DecompressConfigure()`.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompCascadedDecompressAsync(
    const void* compressed_ptr,
    size_t compressed_bytes,
    const void* metadata_ptr,
    size_t metadata_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* uncompressed_ptr,
    size_t uncompressed_bytes,
    cudaStream_t stream);

/**
 * @brief Destroys the metadata object and frees the associated memory.  Must be
 * used to destroy metadata that is generated from
 * nvcompCascadedDecompressConfigure.
 *
 * @param metadata_ptr The pointer to destroy.
 */
void nvcompCascadedDestroyMetadata(void* metadata_ptr);

/**************************************************************************
 *  Cascaded Selector types and API calls
 *************************************************************************/

/**
 * @brief Structure that stores options to run Cascaded Selector
 * NOTE: Minimum values for both parameters is 1, maximum for
 * sample_size is 1024 and is allso limited by the input size:
 *        (sample_size * num_samples) <= input_size
 */
typedef struct
{
  /**
   * @brief The number of elements used in each sample
   * minimum value 1, maximum value 1024
   */
  size_t sample_size;

  /**
   * @brief The number of samples used by the selector
   * minimum value 1
   */
  size_t num_samples;

  /**
   * @brief The seed used for the random sampling
   */
  unsigned seed;

} nvcompCascadedSelectorOpts;

/**
 * @brief Configure the cascaded selector and get the temp memory size needed
 * to run the cascaded selector.
 *
 * @param opts The configuration options for the selector (if null, default
 * values used).
 * @param type The data type of the uncompressed data.
 * @param uncompressed_bytes The size of the uncompressed data in bytes.
 * @param temp_bytes The size of the temporary workspace in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompCascadedSelectorConfigure(
    nvcompCascadedSelectorOpts* opts,
    nvcompType_t type,
    size_t uncompressed_bytes,
    size_t* temp_bytes);

/**
 * @brief Run the cascaded selector to determine the best cascaded compression
 * configuration and estimated compression ratio.
 *
 * @param opts The configuration options for the selector (if null, default
 * values are used).
 * @param type The data type of the uncompressed data.
 * @param uncompressed_ptr The uncompressed data on the device.
 * @param uncompressed_bytes The size of the uncompressed data in bytes.
 * @param temp_ptr The temporary workspace memory on the device
 * @param temp_bytes The size of the temporary workspace in bytes
 * @param format_opts The best cascaded compression configuration (output)
 * @param est_ratio The estimated compression ratio using the configuration
 * (output)
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompCascadedSelectorRun(
    nvcompCascadedSelectorOpts* opts,
    nvcompType_t type,
    const void* uncompressed_ptr,
    size_t uncompressed_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    nvcompCascadedFormatOpts* format_opts,
    double* est_ratio,
    cudaStream_t stream);

/******************************************************************************
 * Batched compression/decompression interface
 *****************************************************************************/

/**
 * @brief Structure that stores the compression configuration
 */
typedef struct
{
  /**
   * @brief The size of each chunk of data to decompress indepentently with
   * Cascaded compression. Chunk size should be in the range of [512, 16384]
   * depending on the datatype of the input and the shared memory size of
   * the GPU being used.
   * Recommended size is 4096.
   * NOTE: Not currently used and a default of 4096 is just used.
   */
  size_t chunk_size;

  /**
   * @brief The datatype used to define the bit-width for compression
   */
  nvcompType_t type;

  /**
   * @brief The number of Run Length Encodings to perform.
   */
  int num_RLEs;

  /**
   * @brief The number of Delta Encodings to perform.
   */
  int num_deltas;

  /**
   * @brief Whether or not to bitpack the final layers.
   */
  int use_bp;
} nvcompBatchedCascadedOpts_t;

// Default options for batched compression
static const nvcompBatchedCascadedOpts_t nvcompBatchedCascadedDefaultOpts
    = {4096, NVCOMP_TYPE_INT, 2, 1, 1};

/**
 * @brief Get temporary space required for compression.
 *
 * NOTE: Batched Cascaded compression does not require temp space, so
 * this will set temp_bytes=0, unless an error is found with the format_opts.
 *
 * @param batch_size The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the
 * batch.
 * @param format_opts The Cascaded compression options and datatype to use.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedCascadedCompressGetTempSize(
    size_t batch_size,
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedCascadedOpts_t format_opts,
    size_t* temp_bytes);

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That
 * is, the minimum amount of output memory required to be given
 * nvcompBatchedCascadedCompressAsync() for each batch item.
 *
 * Chunk size must be limited by the shared memory available on the GPU
 * being used.  In general, it must not exceed 16384, but 4096 bytes is
 * recommended.
 *
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the batch.
 * @param format_opts The Cascaded compression options to use.
 * @param max_compressed_byes The maximum compressed size of the largest chunk
 * (output).
 *
 * @return The nvcompSuccess unless there is an error.
 */
nvcompStatus_t nvcompBatchedCascadedCompressGetMaxOutputChunkSize(
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedCascadedOpts_t format_opts,
    size_t* max_compressed_bytes);

/**
 * @brief Perform batched asynchronous compression.
 *
 * NOTE: Unlike `nvcompCascadedCompressAsync`, a valid compression format must
 * be supplied to `format_opts`.
 *
 * NOTE: The current implementation does not support uncompressed size larger
 * than 4,294,967,295 bytes (max uint32_t).
 *
 * @param[in] device_uncompressed_ptrs Array with size \p batch_size of pointers
 * to the uncompressed partitions. Both the pointers and the uncompressed data
 * should reside in device-accessible memory. The uncompressed data must start
 * at locations with alignments of the data type.
 * @param[in] device_uncompressed_bytes Sizes of the uncompressed partitions in
 * bytes. The sizes should reside in device-accessible memory.
 * @param[in] max_uncompressed_chunk_bytes This argument is not used.
 * @param[in] batch_size Number of partitions to compress.
 * @param[in] device_temp_ptr This argument is not used.
 * @param[in] temp_bytes This argument is not used.
 * @param[out] device_compressed_ptrs Array with size \p batch_size of pointers
 * to the output compressed buffers. Both the pointers and the compressed
 * buffers should reside in device-accessible memory. Each compressed buffer
 * should be preallocated with size at least (8B + the uncompressed size). Each
 * compressed buffer should start at a location with alignment of both 4B and
 * the data type.
 * @param[out] device_compressed_bytes Number of bytes decompressed of all
 * partitions. The buffer should be preallocated in device-accessible memory.
 * @param[in] format_opts The cascaded format options. The format must be valid.
 * @param[in] stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedCascadedCompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t /* max_uncompressed_chunk_bytes */, // not used
    size_t batch_size,
    void* /* device_temp_ptr */, // not used
    size_t /* temp_bytes */,     // not used
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    const nvcompBatchedCascadedOpts_t format_opts,
    cudaStream_t stream);

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The size of the largest chunk in bytes
 * when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedCascadedDecompressGetTempSize(
    size_t num_chunks, size_t max_uncompressed_chunk_bytes, size_t* temp_bytes);

/**
 * @brief Perform batched asynchronous decompression.
 *
 * NOTE: This function is used to decompress compressed buffers produced by
 * `nvcompBatchedCascadedCompressAsync`. Currently it is not compatible with
 * compressed buffers produced by `nvcompCascadedCompressAsync`.
 *
 * @param[in] device_compressed_ptrs Array with size \p batch_size of pointers
 * in device-accessible memory to compressed buffers. Each compressed buffer
 * should reside in device-accessible memory and start at a location with
 * alignment of both 4B and the data type.
 * @param[in] device_compressed_bytes Sizes of the compressed buffers in bytes.
 * The sizes should reside in device-accessible memory.
 * @param[in] device_uncompressed_bytes Sizes of the output uncompressed
 * buffers in bytes. The sizes should reside in device-accessible memory. If the
 * size is not large enough to hold all decompressed elements, the decompressor
 * will set the status specified in \p device_statuses corresponding to the
 * overflow partition to `nvcompErrorCannotDecompress`.
 * @param[out] device_actual_uncompressed_bytes Array with size \p batch_size of
 * the actual number of bytes decompressed for every partitions. This argument
 * needs to be preallocated.
 * @param[in] batch_size Number of partitions to decompress.
 * @param[in] device_temp_ptr This argument is not used.
 * @param[in] temp_bytes This argument is not used.
 * @param[out] device_uncompressed_ptrs Array with size \p batch_size of
 * pointers in device-accessible memory to decompressed data. Each uncompressed
 * buffer needs to be preallocated in device-accessible memory, and start at a
 * location with alignment of the data type.
 * @param[out] device_statuses Array with size \p batch_size of statuses in
 * device-accessible memory. This argument needs to be preallocated. For each
 * partition, if the decompression is successful, the status will be set to
 * `nvcompSuccess`. If the decompression is not successful, for example due to
 * the corrupted input or out-of-bound errors, the status will be set to
 * `nvcompErrorCannotDecompress`.
 * @param[in] stream The cuda stream to operate on.
 */
nvcompStatus_t nvcompBatchedCascadedDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr, // not used
    size_t temp_bytes,           // not used
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream);

/**
 * @brief Asynchronously get the number of bytes of the uncompressed data in
 * every partitions.
 *
 * @param[in] device_compressed_ptrs Array with size \p batch_size of pointers
 * in device-accessible memory to compressed buffers.
 * @param[in] device_compressed_bytes Sizes of the compressed buffers in bytes.
 * The sizes should reside in device-accessible memory.
 * @param[out] device_uncompressed_bytes Sizes of the uncompressed data in
 * bytes. If there is an error when retrieving the size of a partition, the
 * uncompressed size of that partition will be set to 0. This argument needs to
 * be prealloated in device-accessible memory.
 * @param[in] batch_size Number of partitions to check sizes.
 * @param[in] stream The cuda stream to operate on.
 */
nvcompStatus_t nvcompBatchedCascadedGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <thread>

namespace nvcomp
{
  bool get_error_and_clear();
}

using namespace nvcomp;

using T = uint8_t;
using INPUT_VECTOR_TYPE = const std::vector<T>;

#define _INVALID_SIZE 0


class Compressor
{
  uint8_t * d_no_comp_data;
  uint8_t * d_comp_data;
  size_t d_no_comp_size;
  size_t d_comp_size;
  bool error;
  const int _TRY_TIMES=3;

  cudaStream_t stream;

public:
  Compressor(cudaStream_t _stream):
      d_no_comp_size(0),
      d_comp_size(0),
      error(false),
      stream(_stream)
  {

  }

  virtual ~Compressor(){
    release();
  }

  bool is_error_occur(){ return this->error; }

  void prepare_input_data(INPUT_VECTOR_TYPE& input){
    const size_t input_size = sizeof(uint8_t) * input.size();
    if (false == this->check_then_relloc(this->d_no_comp_data, this->d_no_comp_size, input_size)){
      this->set_error("can't malloc input GPU buffer");
      return;
    }

    if (cudaSuccess != cudaMemcpy(this->d_no_comp_data, input.data(), this->d_no_comp_size, cudaMemcpyHostToDevice))
      this->set_error("can't copy to device memory");
  }

  size_t compress_prepared_data(const nvcompBatchedCascadedOpts_t & options){
    if(this->d_no_comp_size == 0){
      printf("no prepared input data\n");
      return _INVALID_SIZE;
    }

    for(int i=0; i< _TRY_TIMES; i++) {
      this->error = false;
      const size_t ret_size = this->try_compress(options);
      if(this->error == false)
        return ret_size;
    }
    return _INVALID_SIZE;
  }

  uint8_t * get_d_comp_data(){ return this->d_comp_data; }

  uint8_t * get_d_no_comp_data(){ return this->d_no_comp_data; }

  size_t decompress(const uint8_t * _d_comp_data){
    for(int i=0; i< _TRY_TIMES; i++) {
      this->error = false;
      const size_t ret_size = this->try_decompress(_d_comp_data);
      if(this->error == false)
        return ret_size;
    }
    return _INVALID_SIZE;
  }

private:
  void set_error(const char * message){
    this->error = true;
    printf("%s\n", message);
  }

  size_t try_decompress(const uint8_t * _d_comp_data){
    CascadedManager manager{nvcompBatchedCascadedDefaultOpts, stream};
    auto decomp_config = manager.configure_decompression(_d_comp_data);

    if(get_error_and_clear()) { // check an error of the last function call
      this->set_error("can't configure decompression");
      return _INVALID_SIZE;
    }

    if (false == this->check_then_relloc(this->d_no_comp_data, this->d_no_comp_size, decomp_config.decomp_data_size)){
      this->set_error("can't configure compression");
      return _INVALID_SIZE;
    }

    if (cudaSuccess != cudaMemset(this->d_no_comp_data, 0, this->d_no_comp_size))
      printf("Failed clear by zero\n");

    manager.decompress(
        this->d_no_comp_data,
        _d_comp_data,
        decomp_config);
    cudaStreamSynchronize(stream);

    const cudaError_t err = cudaStreamSynchronize(stream);
    if(cudaSuccess != err){
      this->set_error("cudaStreamSynchronize return code:");
      printf("%d\n", err);
      return _INVALID_SIZE;
    }

    if(get_error_and_clear()) {
      this->set_error("can't decompress data");
      return _INVALID_SIZE;
    }

    return decomp_config.decomp_data_size;
  }

  bool check_then_relloc(uint8_t*& _d_ptr, size_t & _d_size, const size_t relloc_size){
    if(_d_size > relloc_size) // only up
      return true;
    uint8_t* tmp;
    if (cudaSuccess != cudaMalloc(&tmp, relloc_size)) {
      printf("can't cudaMalloc for GPU buffer");
      return false;
    }
    if(_d_size > 0)
      if (cudaSuccess != cudaFree(_d_ptr))
        printf("cudaFree call failed: (size) %zu\n", _d_size);
    _d_ptr = tmp;
    _d_size = relloc_size;
    return true;
  }

  void release(){
    if(d_no_comp_size > 0)  // avoid free invalid ref
      if (cudaSuccess != cudaFree(this->d_no_comp_data))
        printf("cudaFree call failed: %d\n", __LINE__);
    if(d_comp_size > 0)   // avoid free invalid ref
      if (cudaSuccess != cudaFree(this->d_comp_data))
        printf("cudaFree call failed: %d\n", __LINE__);
  }

  size_t try_compress(const nvcompBatchedCascadedOpts_t & options){
    CascadedManager manager{options, this->stream}; // CudaUtils::check, no throw exception
    if(get_error_and_clear()) { // check an error of the last function call
      this->set_error("can't create CascadedManager");
      return _INVALID_SIZE;
    }

    CompressionConfig comp_config = manager.configure_compression(this->d_no_comp_size); // CudaUtils::check, no throw exception
    if(get_error_and_clear()) { // check an error of the last function call
      this->set_error("can't configure compression");
      return _INVALID_SIZE;
    }

    const size_t size_out = comp_config.max_compressed_buffer_size;
    if(this->check_then_relloc(this->d_comp_data, this->d_comp_size, size_out) == false){
      this->set_error("can't relloc output buffer");
      return _INVALID_SIZE;
    }

    manager.compress(this->d_no_comp_data,
                     this->d_comp_data, comp_config);

    const cudaError_t err = cudaStreamSynchronize(stream);
    if(cudaSuccess != err){
      this->set_error("cudaStreamSynchronize return code:");
      printf("%d\n", err);
      return _INVALID_SIZE;
    }

    if(get_error_and_clear()) {
      this->set_error("can't compress_prepared_data");
      return _INVALID_SIZE;
    }
    const nvcompStatus_t status = *comp_config.get_status();
    if(status != cudaSuccess){
      this->set_error("can't compress_prepared_data");
      printf("compress_prepared_data status: %d\n", status);
      return _INVALID_SIZE;
    }

    const size_t comp_out_bytes = manager.get_compressed_output_size(this->d_comp_data);
    if(get_error_and_clear()) {
      this->set_error("can't get");
      return _INVALID_SIZE;
    }

    return comp_out_bytes;
  }

};

INPUT_VECTOR_TYPE input = {
    0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,34,1,0,2,0,32,3,12,33,34,3,43,4,2,42,41,0,1,1,1,1,0,34,1,0,2,0,32,3,12,33,34,3,43,1,0,1,1,1,1,0,34,1,0,2,0,32,3,12,33,34,3,43,
};

int main()
{
  size_t min_size = input.size()*2;
  nvcompBatchedCascadedOpts_t min_options = {0, NVCOMP_TYPE_UCHAR, 0, 0, false, 0};
  cudaStream_t stream;
  assert(cudaSuccess == cudaStreamCreate(&stream));

  Compressor compressor(stream);

  compressor.prepare_input_data(input);

  if(compressor.is_error_occur()){
    printf("Preparing failed \n"); //todo destructor of stream
    return -1;
  }

  std::thread dummy([]{
    std::cout << "The task requires multithreading... Okey\n";
  });

  // find max compressing scheme
  for(size_t chunk_size = 512; chunk_size < 16384; chunk_size += 512)
    for(int rle = 0; rle < 5; rle++)
        for(int bp = 0; bp < 2; bp++) {
          // No delta without BitPack
          const int max_delta_num = bp == 0 ? 1 : 5;
          for (int delta = 0; delta < max_delta_num; delta++) {
            // No delta mode without delta nums
            const int max_delta_mode = delta == 0 ? 1 : 2;
            for (int delta_mode = 0; delta_mode < max_delta_mode; delta_mode++) {
              printf("chunk_size %zu, rle %d, delta %d, M2Mode %d, bp %d", chunk_size, rle, delta, delta_mode, bp);
              nvcompBatchedCascadedOpts_t options = {chunk_size, nvcomp::TypeOf<T>(), rle, delta, static_cast<bool>(delta_mode), bp};
              const size_t comp_size = compressor.compress_prepared_data(options);
              printf("comp size: %zu\n", comp_size);
              if(compressor.is_error_occur() || comp_size == _INVALID_SIZE)
                continue;
              if(min_size < comp_size)
                continue;

              min_size = comp_size;
              min_options = options;
            }
          }
        }

  printf("min size %d\n", min_size);

  const size_t comp_size = compressor.compress_prepared_data(min_options);
  printf("output compressed size: %zu\n", comp_size);
  if(compressor.is_error_occur() || comp_size == _INVALID_SIZE || comp_size != min_size){
    printf("Compress data failed");
    return -1;
  }

  /*
   * Should there be a copy of the compressed data to host's memory in this step?
   *  The task states nothing about it... so let's decompress the output data
   *  which are in GPU memory.
   */

  const size_t decomp_size = compressor.decompress(compressor.get_d_comp_data());
  // no check: if error was occurred then assert below failed

  // Copy result back to host
  std::vector<T> res(decomp_size);
  cudaMemcpy(
      &res[0], compressor.get_d_no_comp_data(), decomp_size * sizeof(T), cudaMemcpyDeviceToHost);

  const cudaError_t err = cudaStreamDestroy(stream);
  if(cudaSuccess != err)
    printf("cudaStreamDestroy return error code: %d\n", err);

  // Verify correctness
  assert(res == input);

  dummy.join();
}
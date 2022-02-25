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

#include "highlevel/CascadedCommon.h"

#include "DeltaGPU.h"
#include "BitPackGPU.h"
#include "common.h"
#include "type_macros.h"
#include "TempSpaceBroker.h"

#include <cassert>
#include <limits>
extern bool verbose;

namespace nvcomp
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr int const BLOCK_SIZE = 1024;

} // namespace

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

namespace
{

template <typename VALUE>
__global__ void deltaKernel(
    VALUE** const outputPtr,
    const VALUE* const input,
    const size_t* const numDevice,
    const size_t /* maxNum */)
{
  const size_t num = *numDevice;

  if (BLOCK_SIZE * blockIdx.x < num) {
    VALUE* const output = *outputPtr;

    const int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;

    __shared__ VALUE buffer[BLOCK_SIZE + 1];

    if (idx < num) {
      buffer[threadIdx.x + 1] = input[idx];
    }

    if (threadIdx.x == 0) {
      // first thread must do something special
      if (idx > 0) {
        buffer[0] = input[idx - 1];
      } else {
        buffer[0] = 0;
      }
    }

    __syncthreads();

    if (idx < num) {
      output[idx] = buffer[threadIdx.x + 1] - buffer[threadIdx.x];
      printf("%d = %d - %d\n",
             output[idx] , buffer[threadIdx.x + 1] , buffer[threadIdx.x]);
    }
  }
}

template <typename VALUE, typename EXTEND_VALUE>
__global__ void deltaKernelOverflowMode(
    VALUE** const outputPtr,
    const VALUE* const input,
    const size_t* const numDevice,
    const size_t /* maxNum */,
    VALUE* const* const minValOutPtr,
    VALUE* const* const maxValOutPtr
    )
{
  const size_t num = *numDevice;

  if (BLOCK_SIZE * blockIdx.x < num) {
    VALUE* const output = *outputPtr;

    const int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;

    __shared__ VALUE buffer[BLOCK_SIZE + 1];

    if (idx < num) {
      buffer[threadIdx.x + 1] = input[idx];
    }

    if (threadIdx.x == 0) {
      // first thread must do something special
      if (idx > 0) {
        buffer[0] = input[idx - 1];
      } else {
        buffer[0] = 0;
      }
    }

    const VALUE maxValue = **maxValOutPtr;
    const VALUE minValue = **minValOutPtr;

    assert(maxValue >= minValue);
    assert(sizeof(EXTEND_VALUE) > sizeof(VALUE));
    const EXTEND_VALUE sizeOfInterval =
        static_cast<EXTEND_VALUE>(maxValue) -
        static_cast<EXTEND_VALUE>(minValue) + 1;
    __syncthreads();

    if (idx < num) {
      const VALUE next = buffer[threadIdx.x + 1];
      const VALUE prev = buffer[threadIdx.x];

      printf("next %d prev %d\n", next, prev);

      EXTEND_VALUE way = std::abs(
          static_cast<EXTEND_VALUE>(next) -
          static_cast<EXTEND_VALUE>(prev)
          );

      if(way > sizeOfInterval/2) { // overflow way
        if(prev < next)
          way = (minValue - prev) + (next - maxValue) - 1; // 2 < 99 : 1 - 2 + 99 - 100 -1
        else
          way = (next - minValue) + (maxValue - prev) + 1; // 99 < 2 : 2-1 + 100-99 +1
      } else {  //common way
        way = next - prev;
      }

      output[idx] = static_cast<VALUE>(way);
    }

  }
}


} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/
namespace
{

template <typename VALUE>
void deltaLaunch(
    void** const outPtr,
    void const* const in,
    const size_t* const numDevice,
    const size_t maxNum,
    cudaStream_t stream,
    const nvcompCascadedFormatOpts::DeltaOpts::DeltaMode deltaMode,
    void* const* const minValueDevicePtr,
    void* const* const maxValueDevicePtr
    )
{
  VALUE** const outTypedPtr = reinterpret_cast<VALUE**>(outPtr);
  VALUE* const* const minValueDP = reinterpret_cast<VALUE* const*>(minValueDevicePtr);
  VALUE* const* const maxValueDP = reinterpret_cast<VALUE* const*>(maxValueDevicePtr);
  const VALUE* const inTyped = static_cast<const VALUE*>(in);
  if(verbose) printf("deltaLaunch\n");
  const dim3 block(BLOCK_SIZE);
  const dim3 grid(roundUpDiv(maxNum, BLOCK_SIZE));
  bool error_flag = false;

  switch (deltaMode) {
  case nvcompCascadedFormatOpts::DeltaOpts::DeltaMode::NORMAL_DELTA:
    deltaKernel<<<grid, block, 0, stream>>>(
        outTypedPtr, inTyped, numDevice, maxNum);
    break;

  case nvcompCascadedFormatOpts::DeltaOpts::DeltaMode::OVERFLOW_DELTA_FOR_INTERVAL:
    switch (sizeof(VALUE)) {
    case sizeof (int8_t):
      deltaKernelOverflowMode<VALUE, int16_t><<<grid, block, 0, stream>>>(
          outTypedPtr, inTyped, numDevice, maxNum, minValueDP, maxValueDP);
      break;
    default:
      printf("only implement for 8bit\n");
      error_flag = true;
      break;
    }
    break;
  default:
    printf("unknown delta mode option\n");
    error_flag = true;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess || error_flag) {
    throw std::runtime_error(
        "Failed to launch deltaKernel kernel: " + std::to_string(err)
        + " error flag: " + std::to_string(error_flag) );
  }
}


template <typename T>
__global__ void setHeader(
    T* data,
    T** const minValueDevicePtr,
    T** const maxValueDevicePtr,
    unsigned char** const numBitsDevicePtr)
{
  // setup the header and pointers into it
  assert(blockIdx.x == 0);
  assert(threadIdx.x == 0);

  *minValueDevicePtr = data;
  data++;
  *maxValueDevicePtr = data;
  data++;
  *numBitsDevicePtr = reinterpret_cast<unsigned char*>(data);

  printf("setHeader\n");
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

void DeltaGPU::compress(
    void* const  workspace ,
    const size_t workspaceSize,
    const nvcompType_t inType,
    void** const outPtr,
    const void* const in,
    const size_t* const numDevice,
    const size_t maxNum,
    const nvcompCascadedFormatOpts::DeltaOpts::DeltaMode deltaMode,
    cudaStream_t stream
    )
{
  if(verbose) printf("DeltaGPU::compress, mode %d inType %d\n", (int)deltaMode, (int)inType);
  if(deltaMode == nvcompCascadedFormatOpts::DeltaOpts::DeltaMode::OVERFLOW_DELTA_FOR_INTERVAL){
    if(inType != NVCOMP_TYPE_CHAR)
      throw std::runtime_error("Implement only for NVCOMP_TYPE_CHAR");

    const size_t reqWorkSize = requiredWorkspaceSize(maxNum, inType);
    if (workspaceSize < reqWorkSize) {
      throw std::runtime_error(
          "Insufficient workspace size: " + std::to_string(workspaceSize)
          + ", need " + std::to_string(reqWorkSize));
    }
    TempSpaceBroker tempSpace(workspace, workspaceSize);

    void** minValueDevicePtr;
    void** maxValueDevicePtr;
    unsigned char** numBitsDevicePtr;
    tempSpace.reserve(&minValueDevicePtr, 1);
    tempSpace.reserve(&maxValueDevicePtr, 1);
    tempSpace.reserve(&numBitsDevicePtr, 1);
    using T = char;
    T* valueDeviceDtr;
    cudaMalloc((void**)&valueDeviceDtr, 10);

    setHeader<<<1, 1, 0, stream>>>( valueDeviceDtr,
                                  reinterpret_cast<T**>(minValueDevicePtr),
                                   reinterpret_cast<T**>(maxValueDevicePtr),
                                   numBitsDevicePtr);


    //todo setup header

    void* const next_free_ptr = reinterpret_cast<void*>(numBitsDevicePtr + 1);
    const size_t tempSize
      = workspaceSize
        - (static_cast<char*>(next_free_ptr) - static_cast<char*>(workspace)); // -2*sizeof(void*)

    bitpack_helper::calcMinMax(next_free_ptr, tempSize, inType, in, numDevice, maxNum,
                           minValueDevicePtr, maxValueDevicePtr,
                           numBitsDevicePtr, stream);
    NVCOMP_TYPE_ONE_SWITCH(
        inType, deltaLaunch, outPtr, in, numDevice, maxNum, stream, deltaMode,
        minValueDevicePtr, maxValueDevicePtr);
    return;
  }
  NVCOMP_TYPE_ONE_SWITCH(
      inType, deltaLaunch, outPtr, in, numDevice, maxNum, stream, deltaMode, NULL, NULL);
}

size_t DeltaGPU::requiredWorkspaceSize(
    const size_t num, const nvcompType_t  type)
{
  // we need a space for min values, and a space for maximum values
  const size_t bytes = sizeOfnvcompType(type) * bitpack_helper::getReduceScratchSpaceSize(num) * 2;

  return bytes;
}

} // namespace nvcomp

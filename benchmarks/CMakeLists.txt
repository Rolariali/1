file(GLOB EXAMPLE_SOURCES *.cpp *.cu)

if(CMAKE_OLD_CUDA_PACKAGE)
  include_directories(/usr/local/cuda/include)
  set(CUDA_ART "/usr/local/cuda/lib64/libcudart.so")
  set(CUDA_NVML "/usr/local/cuda-11.1/targets/x86_64-linux/lib/stubs/libnvidia-ml.so")
endif()

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  # cut off suffixes
  string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
  string(REPLACE ".cu" "" EXAMPLE_NAME ${EXAMPLE_NAME})
  get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
  add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
  target_link_libraries(${BARE_NAME} PRIVATE nvcomp ${CUDA_ART} ${CUDA_NVML})
endforeach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})

# My

if(CLION)
else()
    set(CUDA_ART "CUDA::cudart")
endif()

if(CMAKE_OLD_CUDA_PACKAGE)
    include_directories(/usr/local/cuda/include)
    set(CUDA_ART "/usr/local/cuda/lib64/libcudart.so")
endif()

add_executable(delta_bp_test delta_bp_test.cpp)
target_link_libraries(delta_bp_test PRIVATE ${CUDA_ART} nvcomp)
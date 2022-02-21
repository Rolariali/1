# My

if(CLION)
else()
    set(CUDA_ART "CUDA::cudart")
endif()

if(CMAKE_OLD_CUDA_PACKAGE)
    include_directories(/usr/local/cuda/include)
    set(CUDA_ART "/usr/local/cuda/lib64/libcudart.so")
endif()

#add_executable(q_selector_test selector_test.cpp)
#target_link_libraries(q_selector_test PRIVATE ${CUDA_ART} nvcomp)
#
#add_executable(q_cascade cascade.cpp)
#target_link_libraries(q_cascade PRIVATE ${CUDA_ART} nvcomp)
#
#add_executable(q_lz4 lz4.cpp)
#target_link_libraries(q_lz4 PRIVATE ${CUDA_ART} nvcomp)
#
#add_executable(q_bp byte_packing.cpp)
#target_link_libraries(q_bp PRIVATE ${CUDA_ART} nvcomp)

add_executable(q_dp delta_packing.cpp)
target_link_libraries(q_dp PRIVATE ${CUDA_ART} nvcomp)

add_executable(q_rlep rle_packing.cpp)
target_link_libraries(q_rlep PRIVATE ${CUDA_ART} nvcomp)

add_executable(delta delta_test.cpp )
target_link_libraries(delta PRIVATE ${CUDA_ART} nvcomp)

#
add_executable(bp byte_packing_test.cpp )
target_link_libraries(bp PRIVATE ${CUDA_ART} nvcomp)
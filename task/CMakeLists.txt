# My

if(CLION)
else()
    set(CUDA_ART "CUDA::cudart")
endif()

if(CMAKE_OLD_CUDA_PACKAGE)
    include_directories(/usr/local/cuda/include)
    set(CUDA_ART "/usr/local/cuda/lib64/libcudart.so")
endif()

add_executable(cascade_test cascade_test.cpp)
target_link_libraries(cascade_test PRIVATE ${CUDA_ART} nvcomp)

add_executable(batch_test batch_test.cpp)
target_link_libraries(batch_test PRIVATE ${CUDA_ART} nvcomp)


#add_executable(q_cascade cascade.cpp
#        #        CascadeWrapper.cpp
#        )
#target_link_libraries(q_cascade PRIVATE ${CUDA_ART} nvcomp)
#
#add_executable(q_lz4 lz4.cpp)
#target_link_libraries(q_lz4 PRIVATE ${CUDA_ART} nvcomp)
#
#add_executable(q_bit byte_packing.cpp)
#target_link_libraries(q_bit PRIVATE ${CUDA_ART} nvcomp)
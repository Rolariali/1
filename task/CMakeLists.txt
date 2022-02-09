# My
add_executable(test test.cpp)
target_link_libraries(test PRIVATE CUDA::cudart nvcomp)

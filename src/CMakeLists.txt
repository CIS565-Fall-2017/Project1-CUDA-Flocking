set(SOURCE_FILES
    "cudaMat4.hpp"
    "glslUtility.hpp"
    "glslUtility.cpp"
    "utilityCore.hpp"
    "utilityCore.cpp"
    "kernel.h"
    "kernel.cu"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )


LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")
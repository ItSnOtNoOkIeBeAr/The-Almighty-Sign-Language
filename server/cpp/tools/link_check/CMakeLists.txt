cmake_minimum_required(VERSION 3.26)
project(link_check LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120)

# TensorRT is not on any default search path; the caller points TENSORRT_ROOT at the unzipped folder.
if(NOT DEFINED TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
endif()
if(NOT TENSORRT_ROOT)
  message(FATAL_ERROR "Set TENSORRT_ROOT to the unzipped TensorRT folder, e.g. -DTENSORRT_ROOT=C:/dev/TensorRT-10.16.1.11")
endif()

find_path(TRT_INCLUDE NvInfer.h PATHS "${TENSORRT_ROOT}/include" NO_DEFAULT_PATH)
find_library(TRT_LIB nvinfer_10 PATHS "${TENSORRT_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT TRT_INCLUDE OR NOT TRT_LIB)
  message(FATAL_ERROR "TensorRT headers/libs not found under '${TENSORRT_ROOT}' (include=${TRT_INCLUDE}, lib=${TRT_LIB})")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(link_check main.cpp)
target_include_directories(link_check PRIVATE "${TRT_INCLUDE}")
target_link_libraries(link_check PRIVATE CUDA::cudart CUDA::nvjpeg "${TRT_LIB}")

cmake_minimum_required(VERSION 3.26)
project(signlang_server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")   # -O3 comes with CMAKE_BUILD_TYPE=Release; no --use_fast_math (exact parity with the Python reference)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

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
file(STRINGS "${TRT_INCLUDE}/NvInferVersion.h" TRT_MAJOR_LINE REGEX "#define TRT_MAJOR_ENTERPRISE")
message(STATUS "TensorRT: ${TENSORRT_ROOT} (${TRT_MAJOR_LINE})")

find_package(CUDAToolkit REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# CPU-only part (network layer, features, decision, config): no CUDA at runtime, so it is also built and tested under AddressSanitizer.
add_library(signlang_net STATIC src/decision.cpp src/features.cpp src/config.cpp src/net.cpp)
target_include_directories(signlang_net PUBLIC src)
target_link_libraries(signlang_net PUBLIC spdlog::spdlog nlohmann_json::nlohmann_json ws2_32)
target_compile_features(signlang_net PUBLIC cxx_std_20)
target_compile_definitions(signlang_net PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)

# GPU part: kernels, TensorRT, nvJPEG, the pipeline. Everything except main() is a library, so the tests link the code the server runs.
add_library(signlang_core STATIC kernels/kernels.cu src/trt.cpp src/jpeg.cpp src/pipeline.cpp)
target_include_directories(signlang_core PUBLIC kernels "${TRT_INCLUDE}")
target_link_libraries(signlang_core PUBLIC signlang_net CUDA::cudart CUDA::nvjpeg "${TRT_LIB}")
set_target_properties(signlang_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
if(MSVC)
  target_compile_options(signlang_net PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/W3 /utf-8>)
endif()

option(SIGNLANG_ASAN "Build signlang_net and the CPU tests with AddressSanitizer (build only signlang_cpu_tests)" OFF)
if(SIGNLANG_ASAN)
  target_compile_options(signlang_net PUBLIC /fsanitize=address /Zi)
  target_link_options(signlang_net PUBLIC /INCREMENTAL:NO)
endif()

add_executable(signlang_server src/main.cpp)
target_link_libraries(signlang_server PRIVATE signlang_core)

enable_testing()
add_executable(signlang_cpu_tests tests/test_decision.cpp tests/test_features.cpp tests/test_net.cpp tests/test_net_robust.cpp)
target_link_libraries(signlang_cpu_tests PRIVATE signlang_net GTest::gtest GTest::gtest_main)
add_test(NAME signlang_cpu_tests COMMAND signlang_cpu_tests)
add_executable(signlang_tests tests/test_kernels.cpp)
target_link_libraries(signlang_tests PRIVATE signlang_core GTest::gtest GTest::gtest_main)
add_test(NAME signlang_tests COMMAND signlang_tests)

# Post-build: copy the TensorRT DLLs next to the exes (vcpkg's toolchain copies spdlog/fmt itself; the CUDA runtime is found on PATH).
foreach(tgt signlang_server signlang_tests signlang_cpu_tests)
  add_custom_command(TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TENSORRT_ROOT}/bin/nvinfer_10.dll" $<TARGET_FILE_DIR:${tgt}>
    COMMENT "copying runtime DLLs next to ${tgt}")
endforeach()

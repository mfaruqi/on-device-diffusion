# Build the benchmark harness against an already-built stable-diffusion.cpp checkout.
#
#   # 1. build sd.cpp itself (see wiki/methods/sdcpp-howto.md), then:
#   cmake -S engines/sdcpp -B $HOME/tools/sd-bench-build -DSDCPP_DIR=$HOME/tools/stable-diffusion.cpp \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build $HOME/tools/sd-bench-build -j8
#
# Links the static libraries from ${SDCPP_DIR}/build, so the harness uses exactly the code that
# sd-cli uses, without compiling sd.cpp a second time. Two executables: sd-bench (plain) and
# sd-bench-nvtx (adds NVTX stage ranges for Nsight Systems).
cmake_minimum_required(VERSION 3.18)
project(sd_bench LANGUAGES C CXX)

set(SDCPP_DIR "" CACHE PATH "stable-diffusion.cpp source checkout, built in ${SDCPP_DIR}/build")
if(NOT SDCPP_DIR)
  message(FATAL_ERROR "set -DSDCPP_DIR=<stable-diffusion.cpp checkout>")
endif()
set(SDCPP_BUILD "${SDCPP_DIR}/build")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_library(NCCL_LIB nccl)

set(SDCPP_STATIC_LIBS
  ${SDCPP_BUILD}/libstable-diffusion.a
  ${SDCPP_BUILD}/ggml/src/libggml.a
  ${SDCPP_BUILD}/ggml/src/ggml-cuda/libggml-cuda.a
  ${SDCPP_BUILD}/ggml/src/libggml-cpu.a
  ${SDCPP_BUILD}/ggml/src/libggml-base.a
  ${SDCPP_BUILD}/thirdparty/libwebm/libwebm.a
  ${SDCPP_BUILD}/thirdparty/libwebp/libwebpmux.a
  ${SDCPP_BUILD}/thirdparty/libwebp/libwebp.a
  ${SDCPP_BUILD}/thirdparty/libwebp/libsharpyuv.a)
foreach(lib ${SDCPP_STATIC_LIBS})
  if(NOT EXISTS ${lib})
    message(FATAL_ERROR "missing ${lib}: build stable-diffusion.cpp first")
  endif()
endforeach()

function(add_bench name)
  add_executable(${name} bench.cpp)
  target_include_directories(${name} PRIVATE ${SDCPP_DIR}/include)
  target_compile_features(${name} PRIVATE cxx_std_17)
  # --start-group lets the linker resolve the circular references between the static archives.
  target_link_libraries(${name} PRIVATE
    -Wl,--start-group ${SDCPP_STATIC_LIBS} -Wl,--end-group
    CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver
    OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
  if(NCCL_LIB)
    target_link_libraries(${name} PRIVATE ${NCCL_LIB})
  endif()
endfunction()

add_bench(sd-bench)
add_bench(sd-bench-nvtx)
target_link_libraries(sd-bench-nvtx PRIVATE CUDA::nvtx3)
target_compile_definitions(sd-bench-nvtx PRIVATE SD_BENCH_NVTX)

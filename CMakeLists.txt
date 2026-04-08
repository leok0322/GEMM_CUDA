cmake_minimum_required(VERSION 3.20)

# 必须在 project() 之前设置，否则 CMake 探测编译器阶段找不到 nvcc
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_COMPILER  "/usr/bin/g++")               # 本机 g++（which g++，版本 11.4.0）

# 必须在 project() 之前设置，CMake 探测 CUDA 编译器时会检查此变量
#set(CMAKE_CUDA_ARCHITECTURES 86)

project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 86)

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# 使用现代 CMake 的 target_include_directories，仅对指定 target 生效，避免全局污染
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
# add_executable(<target> <source1> [source2 ...])
#   sgemm      : target 名称，即最终生成的可执行文件名
#   sgemm.cu   : 主入口源文件（含 main 函数）
#   ${SRC}     : aux_source_directory 收集的 src/ 下所有源文件（各个 kernel 实现）
add_executable(sgemm sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
# PRIVATE：头文件路径仅用于编译 sgemm，不传递给依赖它的其他 target
target_include_directories(sgemm PRIVATE ${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(cuBLAS_sgemm cuBLAS_sgemm.cu)
set_target_properties(cuBLAS_sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_include_directories(cuBLAS_sgemm PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuBLAS_sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(simplest_kernel simplest_kernel.cu)
set_target_properties(simplest_kernel PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_include_directories(simplest_kernel PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(simplest_kernel ${CUDA_LIBRARIES})
cmake_minimum_required(VERSION 3.20)

# 必须在 project() 之前设置，否则 CMake 探测编译器阶段找不到 nvcc
# 若不 set CMAKE_CXX_COMPILER，CMake 自动查找：
#   1. 优先读环境变量 CXX（如 export CXX=/path/to/g++）
#   2. 在 $PATH 中按默认名称依次尝试：c++ -> g++ -> clang++ ...
#   3. Linux 上通常找到 /usr/bin/g++，与显式 set 效果一致
# 显式 set 的意义：PATH 混乱或多版本共存时，锁定到特定编译器
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_COMPILER  "/usr/bin/g++")

# 必须在 project() 之前设置，CMake 探测 CUDA 编译器时会检查此变量
#set(CMAKE_CUDA_ARCHITECTURES 86)

# project(<项目名> LANGUAGES <语言列表>)
#   - 声明项目名称为 GEMM_CUDA，并启用 CXX（C++）和 CUDA 两种语言
#   - CMake 会在此阶段探测并配置对应的编译器（g++ / nvcc）
#   - LANGUAGES 决定了后续 add_executable / add_library 能编译的文件类型
project(GEMM_CUDA LANGUAGES CXX CUDA)

# 将编译命令导出到 compile_commands.json（位于 build/ 目录）
#   - clangd、clang-tidy 等工具依赖此文件做代码分析和补全
#   - 记录每个源文件的完整编译命令（include 路径、宏定义、编译标志等）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(CUDAToolkit REQUIRED) —— 现代 CMake 方式（CMake 3.17+）
#   - 在系统中搜索 CUDA Toolkit，找不到则报错终止（REQUIRED）
#   - 成功后提供现代 IMPORTED target，直接用于 target_link_libraries：
#       CUDA::cudart   : CUDA 运行时库（libcudart）
#       CUDA::cublas   : cuBLAS 库（libcublas）
#       CUDA::cublasLt : cuBLAS Lt 库
#   - 相比旧式 find_package(CUDA)，变量名更规范，路径查找更可靠
#     旧式在新版 CUDA（12+）上 CUDA_CUBLAS_LIBRARIES 可能为空，导致链接报错
find_package(CUDAToolkit REQUIRED)

# include(CheckLanguage)：加载 CMake 内置模块，引入 check_language() 函数
#   - CMake 内置模块位于 /usr/share/cmake-*/Modules/CheckLanguage.cmake
#   - 类似 C++ 中的 #include，不加载则 check_language() 未定义，调用会报错
include(CheckLanguage)

# check_language(CUDA)：探测当前系统是否具备完整可用的 CUDA 编译环境
#   - 实际编译一段最小 CUDA 代码来验证 nvcc 能否正常工作
#   - 结果写入缓存变量 CMAKE_CUDA_COMPILER：
#       有效路径（如 /usr/local/cuda/bin/nvcc）：探测成功
#       NOTFOUND                               ：探测失败
#   - 注意：此处仅做检测，不会因失败而报错终止；
#     若需强制要求，应在之后手动判断：
#       if(NOT CMAKE_CUDA_COMPILER)
#         message(FATAL_ERROR "CUDA compiler not found")
#       endif()
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found")
endif()

# 设置 C++ 语言标准为 C++20，对所有 target 全局生效
#   - 等价于编译器标志 -std=c++20
#   - 配套变量 CMAKE_CXX_STANDARD_REQUIRED ON 可强制要求该标准（否则编译器可能降级）
#   - 配套变量 CMAKE_CXX_EXTENSIONS OFF 可禁用 GNU 扩展（-std=c++20 而非 -std=gnu++20）
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 自动检测当前机器的 GPU Compute Capability，避免硬编码架构号
#   - nvidia-smi 输出格式为 "8.6"，去掉小数点得到 "86"
#   - ERROR_QUIET：nvidia-smi 不存在时静默失败，不中断 cmake 配置
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _GPU_CC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
string(REPLACE "." "" _GPU_CC_NUM "${_GPU_CC}")

if(NOT _GPU_CC_NUM)
    message(FATAL_ERROR "无法检测 GPU 架构，请确认 nvidia-smi 可用")
endif()


# 自定义变量，记录目标 GPU 的计算能力（Compute Capability）版本号
#   - 86 对应 NVIDIA Ampere 架构（如 RTX 3080 / 3090 / A100-SXM4-40GB）
#   - 用于下方 set_target_properties(... CUDA_ARCHITECTURES 86)
#   - nvcc 据此生成对应架构的 PTX / SASS 代码
#   - 常见值：75=Turing(RTX20), 80=Ampere(A100), 86=Ampere(RTX30), 89=Ada(RTX40), 90=Hopper(H100)
# set(CUDA_COMPUTE_CAPABILITY 86)


# 设置全局默认值，新建的 target 自动继承此属性，无需对每个 target 单独 set_target_properties
# CMake 内部将其展开为：-gencode arch=compute_XX,code=[sm_XX,compute_XX]
#   - sm_XX      : 编译为该架构的 SASS 二进制，在当前 GPU 上直接执行，性能最优
#   - compute_XX : 同时保留 PTX 中间码，供未来更新架构的 GPU JIT 编译（向前兼容）
set(CMAKE_CUDA_ARCHITECTURES ${_GPU_CC_NUM})
message(STATUS "检测到 GPU Compute Capability: ${_GPU_CC} -> CUDA_ARCHITECTURES=${_GPU_CC_NUM}")

# add_compile_options(<选项>)：对所有 target 全局追加编译选项
# "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>"
# 是 CMake 生成器表达式（Generator Expression），语法为 $<条件:值>，在生成构建系统时求值：
#
#   $<CONFIG:Debug>              : 当前构建类型是否为 Debug（cmake -DCMAKE_BUILD_TYPE=Debug）
#   $<COMPILE_LANGUAGE:CUDA>     : 当前编译的文件是否是 .cu（CUDA 源文件）
#   $<AND:A,B>                   : A 且 B 同时为真
#   条件为真时，展开为 -G;-src-in-ptx（; 在生成器表达式中表示多个独立选项）
#
# 两个 nvcc 调试标志的含义：
#   -G           : 为 GPU 设备代码生成调试符号，允许 cuda-gdb 单步调试 kernel
#                  副作用：禁用几乎所有 GPU 优化（寄存器分配、指令调度等），性能大幅下降
#   -src-in-ptx  : 在生成的 PTX 汇编中以注释形式插入对应的 .cu 源码行，例如：
#                    //  sum += val * B[k * N + col];   <- .cu 第 43 行
#                    fma.rn.f32  %f2, %f1, %f1, %f3;   <- 对应的 PTX 指令
#                  用途①：cuda-gdb 读取行号映射，断点处可显示 .cu 源码而非裸 PTX
#                  用途②：手动读 PTX 文件时，直接看到每条指令对应哪行源码，无需对照翻找
#                  注意：需与 -G 配合；单独使用时优化器重排导致行号映射不准，意义不大
#                  已移除：PTX 格式只支持 ASCII，源文件含中文注释时 ptxas 报错
#
# 整体语义：仅在 Debug 模式编译 CUDA 文件时附加这两个标志，Release 模式不受影响
# 注意：-src-in-ptx 已移除
#   原因：-src-in-ptx 将源码行以注释形式嵌入 PTX 文件，但 PTX 格式规范只支持 ASCII
#   源文件中含中文注释时，ptxas 解析 PTX 会报：
#     "ptxas fatal: Unexpected non-ASCII character encountered"
#   PTX 没有编码配置选项，是 NVIDIA 的格式限制，无法绕过
#   去掉 -src-in-ptx 后只保留 -G，cuda-gdb 调试仍可用，只损失 PTX 中嵌入源码行的功能
#add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")


add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
# -lineinfo：Release 模式下保留源码行号信息，让 ncu（NVIDIA Nsight Compute）能将性能数据对应回源码行
#   ncu 是独立的外部分析工具，在构建系统之外运行：ncu ./build/gemm 2
#   不需要在 CMakeLists.txt 中集成 ncu 本身，只需保证编译产物含行号信息
#
#   与 -G 的区别：
#     -G（Debug）  ：完整调试信息 + 禁止优化，配合 cuda-gdb 调试用
#                    ncu 在 Debug 下测出的性能是未优化代码的结果，无参考价值
#     -lineinfo    ：仅保留行号，不禁止优化，Release 下 ncu 分析才能反映真实性能
#
#   因此：Debug 用 -G，Release + ncu 性能分析用 -lineinfo，两者分工不同
add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")

# --ptxas-options=-v：让 ptxas（PTX 汇编器）输出每个 kernel 的资源使用统计：
#   - 寄存器数（registers）：影响 GPU 占用率（occupancy）
#   - 共享内存用量（shared memory bytes）
#   - 局部内存用量（local memory bytes，溢出寄存器时产生，性能很差）
# Debug 和 Release 均开启，方便在 Release 下也能看到真实优化后的资源用量
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--ptxas-options=-v>")
add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--ptxas-options=-v>")

# aux_source_directory(<目录> <变量>)：扫描指定目录下的所有源文件，将路径列表存入变量
#   - ${PROJECT_SOURCE_DIR} : CMake 内置变量，指向 project() 所在的根目录（即本文件所在目录）
#   - ${PROJECT_SOURCE_DIR}/src : 即 <项目根>/src/，存放各 kernel 实现的 .cu / .cpp 文件
#   - SRC : 收集结果存入此变量，后续 add_executable(sgemm gemm.cu ${SRC}) 中展开使用
#
# 注意事项：
#   - 只收集已存在的文件，新增源文件后需重新运行 cmake 才能被识别（不会自动重新扫描）
#   - 收集所有"可编译源文件"后缀（.cu .cpp .c .cc 等），不包含 .h / .hpp / .cuh
#     头文件不需要独立编译，只在被 #include 时参与编译，CMake 自动排除
#   - 若目录下有不想编译的源文件需手动排除
#
# 何时需要重新运行 cmake（Reload Project）：
#   需要 ：修改 CMakeLists.txt、新增/删除源文件、修改编译器或架构配置
#   不需要：只修改已有 .cu / .cpp / .cuh 文件内容，直接 make 重新编译即可
#   - 现代 CMake 更推荐显式列出源文件（避免意外编译不相关文件），此处为简化管理而使用
# 使用现代 CMake 的 target_include_directories，仅对指定 target 生效，避免全局污染


aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# add_executable(<target> <source1> [source2 ...])：声明一个可执行文件 target
#   - gemm       : target 名称，即最终生成的可执行文件名（构建后为 ./build/gemm）
#   - gemm.cu    : 主入口源文件，含 main() 函数，负责调用各 kernel 并计时比较
#   - ${SRC}     : 展开为 src/ 下所有 .cu/.cpp 源文件（各 kernel 的具体实现）
#
# CMake 根据文件后缀自动分派编译器：
#   .cu  -> nvcc 编译为设备+主机混合代码  -> gemm.cu.o / sgemm_v1.cu.o ...
#   .cpp -> g++ 编译为纯主机代码          -> utils.cpp.o ...
# 各 .o 独立编译完成后，链接分两步：
#   step1 - nvlink（设备链接）：合并各 .o 中的 GPU 代码，生成 device_linked.o
#   step2 - ld（主机链接）   ：将所有 .o + device_linked.o + 各 .so 链接为最终可执行文件
# nvcc 作为驱动器（driver）按序调用 nvlink 和 ld，CMake 检测到 .cu 源文件时自动选择 nvcc 驱动


add_executable(gemm gemm.cu ${SRC})
# CUDA_ARCHITECTURES 继承自 CMAKE_CUDA_ARCHITECTURES，无需手动指定
#
# target_include_directories(<target> <范围> <路径1> [路径2 ...])
# 等价于给编译器追加 -I<路径>，让源文件中的 #include 能找到对应头文件
#
#   PRIVATE   : 头文件路径仅对 gemm 自身的编译生效
#               若其他 target 链接了 gemm，这些路径不会传递过去
#               对应还有 PUBLIC（自身+传递）和 INTERFACE（仅传递，自身不用）
#
#   ${CUDAToolkit_INCLUDE_DIRS} : CUDA 头文件目录（如 /usr/local/cuda/include）
#                                  使源文件可以 #include <cuda_runtime.h> / <cublas_v2.h> 等
#头文件路径由 CUDA::cudart / CUDA::cublas 的 IMPORTED target 自动携带，无需手动指定，这里显式设置
#   ${PROJECT_SOURCE_DIR}/src   : 项目 src/ 目录，使 gemm.cu 可以 #include "kernel.cuh" 等
#                                  本地 kernel 头文件放在 src/ 下时必须加此路径
target_include_directories(gemm PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)


# target_link_libraries(<target> <库1> [库2 ...])
# 使用现代 IMPORTED target，CMake 自动处理头文件路径、库路径、依赖关系
#
#   CUDA::cudart : libcudart.so —— CUDA 运行时库
#                  cudaMalloc / cudaFree / cudaMemcpy / kernel 启动调度等
#   CUDA::cublas : libcublas.so —— cuBLAS 线性代数库
#                  cublasGemmEx 等 GPU 矩阵运算函数，必须显式链接
#                  不加会报：undefined reference to cublasCreate_v2 / cublasGemmEx
target_link_libraries(gemm CUDA::cudart CUDA::cublas)

# 以上三条指令等价的手动命令行（以架构 86、Release 模式为例）：
#
# 编译阶段（各文件独立）：
#   nvcc -std=c++20 -gencode arch=compute_86,code=[sm_86,compute_86] \
#        -I/usr/local/cuda/include -I./src \
#        -c gemm.cu -o gemm.cu.o
#   nvcc -std=c++20 -gencode arch=compute_86,code=[sm_86,compute_86] \
#        -I/usr/local/cuda/include -I./src \
#        -c src/sgemm_v1.cu -o sgemm_v1.cu.o
#   # ... src/ 下其余源文件同理
#
# 链接阶段（nvcc 驱动，内部依次调用 nvlink -> ld）：
#   nvcc -gencode arch=compute_86,code=[sm_86,compute_86] \
#        gemm.cu.o sgemm_v1.cu.o ... \
#        -lcudart -lcublas \
#        -o gemm
#
# Debug 模式在编译阶段额外追加：-G -src-in-ptx --ptxas-options=-v

add_executable(cuBLAS_gemm cuBLAS_gemm.cu)
# 原写法：set_target_properties(cuBLAS_sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
#   - set_target_properties(<target> PROPERTIES <属性名> <值>)：设置指定 target 的属性
#   - CUDA_ARCHITECTURES：CMake 专用属性，控制 nvcc 的 -gencode 参数，决定编译哪些 GPU 架构
#   - ${CUDA_COMPUTE_CAPABILITY}：之前手动定义的变量，值为 86
# 现已删除：因上方设置了 CMAKE_CUDA_ARCHITECTURES，add_executable 时 target 自动继承，无需重复指定
#set_target_properties(cuBLAS_sgemm PROPERTIES CUDA_ARCHITECTURES 86)
#等价于给 nvcc 传递：
#-gencode arch=compute_86,code=[sm_86,compute_86]
target_include_directories(cuBLAS_gemm PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cuBLAS_gemm CUDA::cudart CUDA::cublas)

add_executable(simplest_kernel simplest_kernel.cu)
target_include_directories(simplest_kernel PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(simplest_kernel CUDA::cudart)



# ════════════════════════════════════════════════════════════════════════════
# 【CLion / CMake 四个操作的含义与对应命令行】
#
# ── Reload CMake Project ─────────────────────────────────────────────────
#   含义：重新执行 CMakeLists.txt，更新构建规则文件（build.ninja / Makefile）
#         不编译任何源文件，不删除已有 .o 文件
#   触发时机：修改 CMakeLists.txt 后必须执行，否则新的编译选项不会写入构建规则
#   命令行：
#     cmake -DCMAKE_BUILD_TYPE=Release \
#           -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
#           -G Ninja \
#           -S /home/liam/cpp_linux/GEMM_CUDA \
#           -B /home/liam/cpp_linux/GEMM_CUDA/cmake-build-release
#
# ── Clean ────────────────────────────────────────────────────────────────
#   含义：删除所有编译产物（.o 文件、可执行文件），保留 CMake 配置（build.ninja）
#         下次 Build 时所有源文件强制重新编译
#   命令行：
#     cmake --build cmake-build-release --target clean
#
# ── Build（增量编译）────────────────────────────────────────────────────
#   含义：只重新编译有变化的源文件（.cu/.cpp 内容变了），其余复用已有 .o
#         Ninja 通过文件时间戳 / 内容哈希判断哪些文件需要重编译
#   注意：CMakeLists.txt 变了但未 Reload，Build 不会感知编译选项的变化
#   命令行：
#     cmake --build cmake-build-release --target gemm -- -j 18
#     │      │      │                   │       │    │   │
#     │      │      │                   │       │    │   └─ 传给底层构建工具（Ninja/Make）的并行参数
#     │      │      │                   │       │    │      -j 18：同时启动 18 个编译进程
#     │      │      │                   │       │    └─ "--"之后的所有参数透传给底层构建工具
#     │      │      │                   │       │       cmake 自身不再解析 "--" 之后的内容
#     │      │      │                   │       └─ 只构建名为 gemm 的 target（对应 add_executable(gemm ...)）
#     │      │      │                   │          不构建 cuBLAS_gemm / simplest_kernel 等其他 target
#     │      │      │                   └─ --target：指定要构建的目标名称
#     │      │      └─ 构建目录：cmake 在此目录下寻找 build.ninja / Makefile
#     │      │         由 cmake -B cmake-build-release 生成（Reload CMake Project 时创建）
#     │      └─ --build：cmake 进入构建模式（而非配置模式）
#     └─ cmake 主程序
#
#   cmake --build 与 make 的关系：
#     cmake --build 是对底层构建工具的封装，本项目生成器为 Unix Makefiles，
#     cmake 读取构建目录后实际执行：gmake -C cmake-build-release gemm -j 18
#     两者本质等价，区别在于：
#       cmake --build：生成器无关（Ninja/Make 均适用），可在任意目录执行（路径作参数）
#       make         ：只对 Make 有效，必须在含 Makefile 的目录下执行
#     make 的三种等价写法：
#       cd cmake-build-release && make gemm -j 18   # cd 进去再执行
#       make -C cmake-build-release gemm -j 18      # -C 指定目录，无需 cd
#       cmake --build cmake-build-release ...        # cmake 内部处理 cd
#
# ── Rebuild ──────────────────────────────────────────────────────────────
#   含义：等价于 Clean + Build，先删除所有产物，再完整重新编译
#         用于解决增量编译状态不一致、头文件依赖混乱等问题
#   命令行：
#     cmake --build cmake-build-release --target clean
#     cmake --build cmake-build-release --target gemm -- -j 18
#
# ── 常见工作流 ──────────────────────────────────────────────────────────
#   修改 CMakeLists.txt  → Reload → Build
#   修改 .cu/.cuh 文件   → 直接 Build（增量编译）
#   编译状态混乱/异常    → Rebuild
#   彻底清除重来         → 删除 cmake-build-release 目录 → Reload → Build
# ════════════════════════════════════════════════════════════════════════════
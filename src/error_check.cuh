#pragma once



// cudaCheck 是项目自定义的错误检查函数，配合 gemm.cu 中的宏：
//   #define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))
// 每次 CUDA API 调用后自动传入源文件名和行号，方便定位出错位置
//
// 【为什么头文件中的函数定义需要 inline】
// 头文件会被多个源文件 #include，若无 inline：
//   runner.cu → #include "error_check.cuh" → 包含 cudaCheck 定义
//   gemm.cu   → #include "error_check.cuh" → 再次包含 cudaCheck 定义
//   链接时两个 .o 都有 cudaCheck 定义 → 报错：multiple definition of `cudaCheck`
//
// 加 inline 后放宽 ODR（One Definition Rule）限制：
//   允许多个翻译单元包含同一函数的定义，链接时只保留一份，不报错
//   注意：现代 C++ 中 inline 的主要作用是放宽 ODR，而非"内联展开"
//         是否内联展开由编译器自行决定，与 inline 关键字无关
inline void cudaCheck(cudaError_t error, const char *file, int line) {
    // cudaSuccess 是枚举值 0，所有 CUDA API 成功时返回此值
    if (error != cudaSuccess) {
        // cudaGetErrorString(error)：将 cudaError_t 枚举值转为可读字符串
        //   如 cudaErrorMemoryAllocation -> "out of memory"
        //      cudaErrorInvalidDevice    -> "invalid device ordinal"
        // 若不调用此函数，error 只是一个整数，无法直观看出错误原因
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
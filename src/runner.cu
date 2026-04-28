// 12_kernel_double_buffering.cuh 的引用链路（间接 include）：
//   runner.cu
//     → #include "kernels.cuh"          （汇总头文件，集中 include 所有 kernel）
//         → #include "kernels/12_kernel_double_buffering.cuh"
//
// 预处理器按链路依次展开，最终 12_kernel_double_buffering.cuh 的内容
// 原地粘贴进 runner.cu 的翻译单元，编译器看到的是一个合并后的完整文件
//
// CMakeLists.txt 中 target_include_directories(gemm PRIVATE ... ${PROJECT_SOURCE_DIR}/src)
// 的作用正是让预处理器能正确找到 "kernels.cuh"（位于 src/）以及其中的 "kernels/xxx.cuh"
// 等价于编译命令追加 -I/path/to/src，缺少此路径则 #include "kernels.cuh" 会报找不到文件
#include "kernels.cuh"
// runner.cu 包含自己的头文件 runner.cuh，目的是让编译器做自洽性检查：
//   runner.cuh 中是函数声明（declaration），runner.cu 中是函数定义（definition）
//   同时 include 后，若定义的参数类型或返回值与声明不一致，编译时立即报错
//   若不包含，runner.cu 能正常编译，但声明与定义的不匹配只能在链接阶段才能发现，更难调试
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include "error_check.cuh"


#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))  // 需要在cudaCheck定义之后才能定义宏



float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }



void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N) {
  // struct timeval time {}：Linux 系统结构体，{} 零初始化，包含两个字段：
  //   tv_sec  : 自 1970-01-01 00:00:00 至今的秒数（精度：秒）
  //   tv_usec : 当前秒内的微秒偏移量（精度：微秒，范围 0~999999）
  struct timeval time {};

  // gettimeofday(&time, nullptr)：获取当前时间填入 time
  //   第二个参数为时区信息，传 nullptr 表示忽略
  gettimeofday(&time, nullptr);

  // srand(time.tv_usec)：以微秒级时间戳作为随机种子初始化随机数生成器
  //   不用 srand(time(NULL))：time(NULL) 精度只到秒，同一秒内多次调用种子相同，
  //   生成的随机数序列完全一致；tv_usec 精度到微秒，碰撞概率大幅降低
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    // 类型推导：
    //   rand()                  : int
    //   rand() % 5              : int（int % int = int）
    //   (float)(rand() % 5)     : float（显式强转）
    //   0.01                    : double（浮点字面量默认 double，float 需写 0.01f）
    //   0.01 * (rand() % 5)     : double（double * int，int 隐式提升为 double）
    //   (float)(...) + 0.01*(...): double（float + double，float 提升为 double）
    //   赋给 float tmp           : double 隐式截断为 float（向下转换，有精度损失）
    //
    // 隐式提升 vs 向下转换：
    //   隐式提升（promotion） ：低精度 → 高精度，无损失（int→double, float→double）
    //   向下转换（narrowing）  ：高精度 → 低精度，有精度损失（double→float）
    //   两者方向相反，不可混淆
    //
    // double 和 float 没有继承关系，C++ 基本类型之间不存在子类概念
    // 隐式提升来自 C++ 算术转换规则（arithmetic conversion）：
    //   运算时低精度操作数自动转换为高精度：
    //   long double > double > float > long > int > short > char
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    // -1. 也是 double 字面量，tmp * (-1.) 结果为 double，赋回 float tmp 时截断
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

// 将行主序一维数组 A（M×N 矩阵）以可读格式写入文件流 fs
// 输出格式类似 MATLAB/numpy：
//   [ 1.23,  4.56;       ← 第 0 行，; 表示行结束
//     7.89,  0.12]       ← 最后一行，] 表示矩阵结束
// std::ofstream &fs：引用传参，避免拷贝，直接操作调用方的文件流
void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  // C89 风格：变量必须在函数开头声明，不能在 for 里声明
  // 现代 C++ 写法应为 for(int i = 0; ...)，作用域限制在循环内，更安全
  // int i;
  // std::setprecision(2)：设置浮点数有效位数为 2 位小数
  // std::fixed：固定小数点格式（如 3.14），而非科学计数法（如 3.14e+00）
  // 这两个是 io 操纵符（manipulator），只影响流（stream）的 << 输出格式，设置后持续生效
  //   不影响变量本身的值、printf 格式、普通赋值和计算
  //   本质是修改流对象内部的格式状态，与数据无关
  fs << std::setprecision(2)
     << std::fixed;
  fs << "[";
  // 按行主序遍历一维数组，i 从 0 到 M*N-1
  for (int i = 0; i < M * N; i++) {
    // std::setw(5)：设置下一个输出项的最小字段宽度为 5
    //   单位是字符数（字符个数），不足 5 个字符时左侧补空格对齐
    //   例：3.14 → " 3.14"（1个空格+4个字符=5），-3.14 → "-3.14"（恰好5个字符）
    //   超过 5 个字符时不截断，按实际宽度输出
    // 注意：setw 只对紧跟的下一次输出生效，不像 setprecision 持续有效
    if ((i + 1) % N == 0)
      // (i+1) % N == 0：当前元素是该行最后一个，行尾不加逗号
      fs << std::setw(5) << A[i];
    else
      // 非行尾元素：写值后加 ", " 分隔
      fs << std::setw(5) << A[i] << ", ";

    if ((i + 1) % N == 0) {
      // 行尾处理：除最后一行外，写 ";\n" 表示行结束（MATLAB 风格）
      // i+1 < M*N：还有下一行，写分号换行
      // i+1 == M*N：最后一行，不写分号，由循环外的 "]" 收尾
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

// 对比自写 kernel 输出（matOut）与 cuBLAS 参考结果（matRef）逐元素验证正确性
bool verify_matrix(float *matRef, float *matOut, int N) {
  // diff 用 double 而非 float：两个 float 相减可能损失精度，double 中间计算更准确
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    // std::fabs()：取浮点数绝对值，在 <cmath> 中有三个重载版本：
    //   float       std::fabs(float)
    //   double      std::fabs(double)
    //   long double std::fabs(long double)
    //   编译器在编译期根据参数类型选择对应的重载版本（此处参数为 float，选 float 版本）
    //   这是函数重载（overload）而非函数模板（template）：
    //     函数重载：多份独立实现，不同类型可有不同底层指令
    //     函数模板：一份代码自动实例化，逻辑必须相同
    //   float/double/long double 底层浮点指令不同，故用重载而非模板


    //   C 风格的 fabsf() 只接受 float、fabs() 只接受 double，类型写死不够灵活
    //   cmath 中 float 版本的实际实现：
    //     inline _GLIBCXX_CONSTEXPR float fabs(float x) { return __builtin_fabsf(x); }
    // #ifndef __CORRECT_ISO_CPP_MATH_H_PROTO
    // // 条件编译守卫，防止与 C 的 math.h 中的 fabs 定义冲突
    // // 只在符合 ISO C++ 标准的环境下才启用这套重载
    //
    // inline
    // // 建议编译器将函数体直接展开到调用处，避免函数调用开销
    // // 对于这种极简函数（只有一行），inline 几乎必然被采纳
    //
    // _GLIBCXX_CONSTEXPR
    // // GCC 标准库的宏，展开为 constexpr
    // // 表示此函数可在编译期求值（如 constexpr float x = fabs(-1.0f)）
    //
    // __builtin_fabsf(__x)
    // // __builtin_* 是 GCC 内置函数（builtin），不是普通 C/C++ 函数
    // // 编译器直接将其翻译为硬件浮点指令，如 x86 的 FABS 或 SSE 的 ANDPS
    // // 没有函数调用，没有跳转，性能最优
    // // （x86: ANDPS/FABS），无函数调用开销，cmath 只是薄薄的包装层，实现在编译器内部

    // 不能直接比较 matRef[i] == matOut[i]：浮点运算存在舍入误差，完全相等几乎不可能
    // 改为比较绝对误差：|matRef[i] - matOut[i]| < 0.01 视为正确
    diff = std::fabs(matRef[i] - matOut[i]);
    // isnan(diff)：检测 NaN（kernel 计算出现除零、溢出等非法结果时产生）
    // diff > 0.01：误差超过阈值，认为结果发散
    if (isnan(diff) || diff > 0.01) {
      // %5.2f：浮点格式说明符，% 起始，5 总宽度（不足左补空格），.2 小数点后2位，f 浮点数
      //   作用：三列数字宽度一致，对齐排列，便于对比阅读
      //   例：Should  3.14, Is  3.17 (Diff  0.03)
      // %d：十进制整数格式说明符，输出出错元素的下标 i
      // \n：转义字符，表示换行，与 %d 无关，是独立的字符串转义
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // cuBLAS 内部使用列主序（column-major），而本项目矩阵是行主序（row-major）
  // 行主序的 A(M×K) 在列主序视角下等价于 A^T(K×M)，直接传入会算错
  // 利用转置等价公式绕过：(A×B)^T = B^T × A^T
  //   列主序下计算 B^T × A^T，等价于行主序下计算 A × B
  //   实现方式：交换 A/B 的传入顺序，将 N/M 对调，cuBLAS 内部自动处理转置
  //
  // cublasGemmEx 函数签名（简化）：
  //   cublasStatus_t cublasGemmEx(
  //     cublasHandle_t handle,
  //     cublasOperation_t transa,  // 对第一个矩阵的操作：CUBLAS_OP_N=不转置
  //     cublasOperation_t transb,  // 对第二个矩阵的操作：CUBLAS_OP_N=不转置
  //     int m,                     // 第一个矩阵的行数（列主序视角）→ 传 N
  //     int n,                     // 第二个矩阵的列数（列主序视角）→ 传 M
  //     int k,                     // 内维度
  //     const void *alpha,         // 缩放系数 α（传指针）
  //     const void *A,             // 第一个矩阵 → 传 B（交换顺序）
  //     cudaDataType_t Atype,      // 第一个矩阵数据类型：CUDA_R_32F = float32
  //     int lda,                   // leading dimension：列主序下第一个矩阵的行数 → 传 N
  //     const void *B,             // 第二个矩阵 → 传 A（交换顺序）
  //     cudaDataType_t Btype,      // 第二个矩阵数据类型：CUDA_R_32F = float32
  //     int ldb,                   // leading dimension：列主序下第二个矩阵的行数 → 传 K
  //     const void *beta,          // 缩放系数 β（传指针）
  //     void *C,                   // 输出矩阵
  //     cudaDataType_t Ctype,      // 输出矩阵数据类型：CUDA_R_32F = float32
  //     int ldc,                   // leading dimension：输出矩阵的行数 → 传 N
  //     cublasComputeType_t computeType, // 计算精度：CUBLAS_COMPUTE_32F = 全程 float32
  //     cublasGemmAlgo_t algo      // 算法选择：CUBLAS_GEMM_DEFAULT_TENSOR_OP = 自动选最优
  //   )
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasBF16(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // This runs cuBLAS with mixed precision (performing the mul with operands
  // downcast to bf16), which is ~4x faster
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasTF32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // This runs cuBLAS with mixed precision (performing the mul with operands
  // downcast to bf16), which is ~4x faster
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void run_gemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  // blockDim(32, 32)：每个 block 含 32×32=1024 个线程，一个线程负责 C 矩阵的一个元素
  //
  // 【线程编号展开规则】
  //   block 内线程按 x 优先线性化：linear_id = threadIdx.x + threadIdx.y * blockDim.x
  //   blockDim(32,32) 时展开如下：
  //     linear_id  0~ 31 → (threadIdx.x=0..31, threadIdx.y=0)  = warp 0
  //     linear_id 32~ 63 → (threadIdx.x=0..31, threadIdx.y=1)  = warp 1
  //     ...
  //     linear_id 992~1023→ (threadIdx.x=0..31, threadIdx.y=31) = warp 31
  //   展开规则与 C 矩阵访存是否连续无关，只描述线程如何分配到 warp
  //   32×32=1024 是为了凑满每个 block 的线程数上限，充分利用 SM 资源
  //
  // 为什么 x 维度选 32（而非 blockDim(1,32)）：
  //   blockDim(1,32) 时同一 warp 内 threadIdx.y=0..31，threadIdx.x 全为 0
  //   → x 全相同，y 连续
  //   访问 A[x*K+i]：x 相同 → 32 线程访问同一地址 → 广播
  //   访问 B[i*N+y]：y 连续 → 地址连续 → coalesced ✓
  //   访问 C[x*N+y]：x 相同 y 连续 → 地址连续 → coalesced ✓
  //   访存模式尚可，但核心问题是 occupancy 不足：
  //   每 block 只有 1 个 warp，Ampere 每 SM 最多驻留 16 个 block
  //   → 16 block × 1 warp = 16 warp，occupancy = 16/32 = 50%
  //   访存等待时无足够 warp 可切换，SM 空转，延迟无法隐藏
  //   blockDim(32,32) 每 block 32 个 warp，1 个 block 填满 SM，occupancy = 100%
  //
  // 为什么 y 维度选 32（而非 blockDim(32,1)）：
  //   blockDim(32,1) 时同一 warp 内 threadIdx.x=0..31，threadIdx.y=0
  //   → x 连续，y 全相同
  //   访问 A[x*K+i]：x 连续 → 地址间距 K*sizeof(float) → 散乱，32 次独立事务（最差）
  //   访问 B[i*N+y]：y 全相同 → 32 线程访问同一地址 → 广播
  //   访问 C[x*N+y]：x 连续 y 相同 → 地址间距 N*sizeof(float) → 散乱，32 次独立事务（最差）
  //   访存模式与 blockDim(32,32) 的 warp 0 完全一样，区别只在 occupancy：
  //   每 block 只有 1 个 warp，Ampere 每 SM 最多驻留 16 个 block
  //   → 16 block × 1 warp = 16 warp，occupancy = 16/32 = 50%，访存延迟无法充分隐藏
  //   blockDim(32,32) 每 block 32 个 warp，1 个 block 填满 SM，occupancy = 100%
  dim3 blockDim(32, 32);

  // gridDim：用足够多的 block 覆盖整个 C 矩阵（M×N）
  //
  // 【block 编号展开规则】
  //   grid 内 block 同样按 x 优先线性化：linear_id = blockIdx.x + blockIdx.y * gridDim.x
  //   每个 block 覆盖 C 矩阵 32×32 的子块，block(bx,by) 负责：
  //     行范围：by*32 ~ by*32+31
  //     列范围：bx*32 ~ bx*32+31
  //   需要 ⌈M/32⌉×⌈N/32⌉ 个 block 才能覆盖整个 C 矩阵
  //   CEIL_DIV 向上取整：保证矩阵边缘（尺寸不是 32 整数倍时）也被覆盖
  //   多余的线程由 kernel 内 if(x<M && y<N) 过滤，不做计算
  //   例：M=N=1024 -> gridDim=(32,32)，总线程数=1024×1024=C 矩阵元素总数
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  gemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
  cudaCheck(cudaGetLastError());
}

void run_gemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // blockDim(32*32=1024)：每个 block 1024 个线程，1D 排列
  // 模板参数 <32> 即 BLOCKSIZE=32，kernel 内用 /32 和 %32 将 threadIdx.x 映射为 2D 坐标
  //   threadIdx.x / 32 → block 内行偏移，范围 0~31，共 32 行
  //   threadIdx.x % 32 → block 内列偏移，范围 0~31，共 32 列
  // 因此每个 block 恰好覆盖 C 的一个 32×32 tile（32行 × 32列 = 1024个元素）
  //
  // gridDim(CEIL_DIV(M,32), CEIL_DIV(N,32))：
  //   x 方向 CEIL_DIV(M,32) 个 block，覆盖所有 M 行
  //   y 方向 CEIL_DIV(N,32) 个 block，覆盖所有 N 列
  //   grid 内所有 block 的 tile 拼合后覆盖整个 M×N 的 C 矩阵
  //   超出实际矩阵范围的线程由 kernel 内 if(cRow<M && cCol<N) 过滤
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  gemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
  cudaCheck(cudaGetLastError());
}



void run_gemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C, int deviceIdx, bool& record) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);


  cudaDeviceProp deviceProp {};
  cudaCheck(cudaGetDeviceProperties_v2(&deviceProp, deviceIdx));

  cudaFuncAttributes funcAttributes {};
  // reinterpret_cast<const void*>：强制将 __global__ 函数符号转换为 const void*
  // cudaFuncGetAttributes 的第二个参数类型是 const void*，需要显式转换

  // gemm_shared_mem_block<32> 是设备端符号（device symbol），不是普通函数指针：
  //   不是普通函数指针void(*)(int, int, int, float, const float*, const float*, float, float*)
  //   CUDA 编译器给 __global__ 函数赋予了一个特殊的不透明类型，不是标准 C++ 函数指针。ReSharper 因此显示为 __resharper_unknown_type
  //   编译后成为设备端符号表中的一个条目，主机端持有的是指向该符号的句柄
  //   CUDA 运行时通过此句柄在设备显存中查找 kernel 实际入口地址
  //   不能用 cudaGetSymbolAddress 获取（那是给 __device__ 变量用的，不适用于 __global__ 函数：
  //      __device__ float d_data[1024];
  //      void* ptr;
  //      cudaGetSymbolAddress(&ptr, d_data);  // ✓ 获取设备端变量地址
  //   GPU 函数地址在设备显存中，与 CPU 地址空间隔离，CUDA 故意不暴露实际地址

  // 为什么不能用 static_cast<const void*>：
  //   static_cast 要求类型之间有标准定义的转换关系
  //   函数指针 → void* 在 C++ 标准中不是合法的 static_cast 转换
  //   即使是普通函数指针 void(*)(int,...) 也同样不能 static_cast 为 void*：
  //     void foo(int){}; static_cast<void*>(foo)  → 编译错误
  //     reinterpret_cast<void*>(foo)               → 合法
  //   函数指针和数据指针（void*）在 C++ 标准中属于完全不同的类型体系，static_cast 无法跨越

  // reinterpret_cast 可以：
  //   强制重新解释指针的二进制表示，不检查类型关系
  //   C 风格转换 (const void*) 在此场景等价于 reinterpret_cast，两者均可
  //   CUDA 文档要求用此方式传入 kernel 函数地址
  cudaCheck(cudaFuncGetAttributes(&funcAttributes, reinterpret_cast<const void*>(gemm_shared_mem_block<32>)));

  if (record) {
    std::string fileNme {"benchmark_results/kernel_3_properties.txt"};
    std::ofstream file;
    file.open(fileNme);

    file << "defore cudaFuncSetAttribute,\n";
    file << "device properties are listed as below: \n";
    file << "deviceProp.sharedMemPerMultiprocessor: ";
    // 运算优先级：/ 和 << 都是二元运算符，/ 优先级（5）高于 <<（7，按 C++ 标准数字越小越高）
    // 因此 deviceProp.sharedMemPerMultiprocessor / (1<<10) 先算除法，再输出到 file
    // (1<<10) = 1 * 2^10 = 1024，整除后得到 KiB 数值
    // 括号是必须的：若写 1<<10 而不加括号，<< 会被解析为流插入运算符，语义完全错误
    //   file << x / 1 << 10  →  (file << (x/1)) << 10  →  先输出 x，再输出整数 10，结果错误
    // 加括号后：file << (x / (1<<10))  →  先计算 1<<10=1024，再除，再输出，正确
    file << deviceProp.sharedMemPerMultiprocessor / (1<<10);
    file << " KiB";
    file << "\n";
    file << "deviceProp.reservedSharedMemPerBlock: ";
    file << deviceProp.reservedSharedMemPerBlock / (1<<10);
    file << " KiB";
    file << "\n";
    file << "deviceProp.sharedMemPerBlock: ";
    file << deviceProp.sharedMemPerBlock / (1<<10);
    file << " KiB";
    file << "\n";
    file << "deviceProp.sharedMemPerBlockOptin: ";
    file << deviceProp.sharedMemPerBlockOptin  / (1<<10);
    file << " KiB";
    file << "\n";
    file << "deviceProp.regsPerBlock: ";
    file << deviceProp.regsPerBlock;
    file << "\n";
    file << "deviceProp.regsPerMultiprocessor: ";
    file << deviceProp.regsPerMultiprocessor;
    file << "\n\n";
    file << "kernel_3 properties are listed as below: \n";
    file << "funcAttributes.sharedSizeBytes: ";
    file << funcAttributes.sharedSizeBytes / (1<<10);
    file << " KiB";
    file << "\n";
    file << "funcAttributes.maxDynamicSharedSizeBytes: ";
    file << funcAttributes.maxDynamicSharedSizeBytes / (1<<10);
    file << " KiB";
    file << "\n";
    file << "funcAttributes.preferredShmemCarveout: ";
    file << funcAttributes.preferredShmemCarveout;
    file << "\n";
    file << "funcAttributes.localSizeBytes: ";
    file << funcAttributes.localSizeBytes;
    file << "\n";
    file << "funcAttributes.numRegs: ";
    file << funcAttributes.numRegs;
    file << "\n\n";


    // ════════════════════════════
    // 【cudaFuncSetAttribute 详解：编译期 vs 运行期，两个属性的作用与关系】
    //
    // ── 背景：SM 上的三类片上资源 ─────────
    //
    //   GPU 的每个 SM（Streaming Multiprocessor）上有三类片上资源，总量由硬件固定：
    //
    //   ① 寄存器（Register File）
    //      总量由 deviceProp.regsPerMultiprocessor 决定（典型值 65536 个 32-bit 寄存器）
    //      分配粒度：线程级别（每线程需要多少寄存器由编译器静态分析决定）
    //
    //      一个线程占用多个寄存器的情况：
    //        kernel 中每个局部变量、中间计算结果都需要一个寄存器存放
    //        例如：
    //          float a = A[i];        // 1 个寄存器存 a
    //          float b = B[i];        // 1 个寄存器存 b
    //          float acc = 0.0f;      // 1 个寄存器存累加器
    //          float tmp = a * b;     // 编译器可能复用，也可能再分配 1 个
    //        kernel_3 的 ptxas 输出：Used 31 registers
    //        → 每个线程同时持有 31 个寄存器，block 内 1024 个线程共占
    //          31 × 1024 = 31744 个寄存器
    //        → SM 有 65536 个寄存器，最多容纳 floor(65536/31744) = 2 个 block 并发
    //        寄存器是占用率（occupancy）的主要瓶颈之一：
    //          每线程寄存器越多 → 单 block 消耗越大 → SM 能并发的 block 越少
    //          这也是高级 kernel 做"寄存器 tiling"（register tiling）的动机：
    //          把更多中间结果放在寄存器而非反复读 SMEM，以计算换访存，
    //          但需权衡寄存器增多导致的 occupancy 下降
    //      不够时：spill 到 Local Memory（每线程私有，物理上是 HBM 的一段地址，
    //              经 L1/L2 cache，延迟从 0 cycles 上升到 100~300+ cycles）
    //      编译时可见，kernel_3 实际 ptxas 输出：
    //        ptxas info: Used 31 registers, used 1 barriers, 8192 bytes smem, 400 bytes cmem[0]
    //        ptxas info: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    //
    //        Used 31 registers：每线程使用 31 个寄存器
    //          由编译器静态分析所有局部变量、中间计算结果后确定
    //          影响 occupancy：31 × 1024（线程/block）= 31744 个寄存器/block
    //                         SM 共 65536 个 → 最多 floor(65536/31744) = 2 个 block 并发
    //
    //        used 1 barriers：kernel 内使用了 1 次 __syncthreads()
    //          barrier 是 block 内所有线程的同步点，确保 shared memory 加载完成后再计算
    //          barrier 数量也是编译期静态确定的资源，影响 SM 调度
    //
    //        8192 bytes smem：单个 block 的静态 shared memory 用量 = 8 KiB
    //          对应 kernel_3 中两个 __shared__ tile：
    //            __shared__ float As[32][32]  = 32*32*4 = 4096 bytes
    //            __shared__ float Bs[32][32]  = 32*32*4 = 4096 bytes
    //            合计 = 8192 bytes = 8 KiB
    //          以单个 block 为单位统计，每个 block 启动时固定占用此量
    //          SM 上同时运行 N 个 block，物理 SMEM 消耗 = N × 8192 bytes
    //          即 funcAttributes.sharedSizeBytes 的值
    //
    //        400 bytes cmem[0]：constant memory 用量
    //          cmem[0] 是 constant memory bank 0，专门存放 kernel 函数参数
    //          kernel 参数（M, N, K, alpha, beta, 指针等）在调用时由驱动自动打包写入
    //          所有线程读同一个地址时硬件广播，零额外开销
    //
    //        0 bytes stack frame：GPU 栈帧为 0
    //          GPU kernel 不应有递归或动态大小的局部数组，否则需要栈空间
    //          栈帧非 0 会显著降低性能（需要额外内存分配）
    //
    //        0 bytes spill stores / 0 bytes spill loads：无寄存器溢出
    //          spill stores：寄存器不够，将数据写出到 Local Memory 的字节数
    //          spill loads ：后续从 Local Memory 读回的字节数
    //          两者均为 0 说明 31 个寄存器完全够用，无溢出，性能最优
    //
    //   ② Shared Memory（SMEM）
    //      与 L1 Cache 共享同一块片上 SRAM，两者按比例切分
    //      分配粒度：block 级别（每个 block 占用固定量，SM 上多个 block 共用总量）
    //      不够时：kernel launch 失败（cudaErrorInvalidConfiguration），不是运行时崩溃
    //      编译时可见：ptxas 输出 "N bytes smem"（静态部分）
    //
    //   ③ L1 Cache
    //      与 Shared Memory 共享同一块片上 SRAM，由驱动/设置决定比例
    //      Local Memory 的 spill 数据也走 L1 缓存（localL1CacheSupported=1 时）
    //
    //   三类资源共同决定 occupancy（SM 上能并发的 warp 数 / 理论最大 warp 数）：
    //     floor(regsPerMultiprocessor / (每线程寄存器数 × block线程数))
    //     floor(sharedMemPerMultiprocessor / 每个block的smem用量)
    //     硬件 block 数上限（sm_86 = 16）
    //   以上三个约束取最严格的，就是实际能并发的 block 数


    // ── 四个字段的性质总结 ──────────────────────────────────────────────────
    //
    //   字段                               性质        含义
    //   ─────────────────────────────────────────────────────────────────────
    //   sharedMemPerMultiprocessor         硬件上限    SM 物理 SRAM 最大可给 SMEM 的量
    //                                                  （carveout=100% 时）
    //   funcAttributes.maxDynamicSharedSizeBytes 软件上限  kernel 被允许申请的 dynamic SMEM 上限
    //                                                  不是实际占用，是申请配额的天花板
    //   sharedMemPerBlock                  软件默认值  不 opt-in 时的保守上限（历史兼容值）
    //                                                  与物理分区无关，任何设置都不会改变它
    //   funcAttributes.sharedSizeBytes     真实占用    编译器静态确定的 __shared__ 变量大小
    //                                                  每个 block 启动时的实际物理占用量
    //   ─────────────────────────────────────────────────────────────────────
    //   注：dynamic SMEM 的真实占用 = <<<>>> 第三个参数实际传入值（本 kernel = 0）
    //       L1 大小无法通过以上任何字段直接读取，只能由 ncu profiler 间接观测
    //
    // ── deviceProp 中 shared memory 相关字段（硬件固定，任何调用都不会改变）──
    //
    //   deviceProp.sharedMemPerMultiprocessor = 100 KiB（本机 sm_86）
    //     硬件上限：SM 物理 SRAM 在 carveout=100% 时全部给 SMEM 的最大量
    //     不反映当前实际 L1/SMEM 分区，设置 carveout=MaxL1 后此值仍为 100 KiB
    //
    //   deviceProp.sharedMemPerBlock = 48 KiB
    //     软件默认值：不调用 cudaFuncSetAttribute 时单 block 的 SMEM 上限
    //     保守的历史兼容值，与物理分区无关，任何设置都不会改变它
    //
    //   deviceProp.sharedMemPerBlockOptin = 99 KiB
    //     opt-in 路径下单 block 的 SMEM 理论上限
    //     主动申请后（见下文）才能突破 48 KiB 默认值，不能超过此值
    //
    //   deviceProp.reservedSharedMemPerBlock = 1 KiB
    //     系统/驱动保留的 SMEM，不对用户代码开放
    //     实际可用 = sharedMemPerBlockOptin - reservedSharedMemPerBlock = 98 KiB
    //
    // ── cudaFuncAttributes 中相关字段（per-kernel运行时属性，可被修改）────
    //
    //   funcAttributes.sharedSizeBytes = 8 KiB（本 kernel）
    //     真实占用：编译期由 ptxas 静态确定的 __shared__ 变量大小
    //     每个 block 启动时固定占用此量，是唯一能直接反映物理 SMEM 占用的字段
    //     ptxas 输出：Used N registers, M bytes smem ← M 就是 sharedSizeBytes
    //
    //   funcAttributes.maxDynamicSharedSizeBytes
    //     软件上限：kernel 被允许通过 <<<grid, block, dynamicSmem>>> 申请的动态 SMEM 上限
    //     不是实际占用——kernel_3 第三个参数缺省(=0)，实际动态占用 = 0，不是 40/90 KiB
    //     默认值 = sharedMemPerBlock - sharedSizeBytes = 48 - 8 = 40 KiB
    //     可通过 cudaFuncAttributeMaxDynamicSharedMemorySize 提升到最多 91 KiB（见下）
    //
    //   funcAttributes.preferredShmemCarveout
    //     偏好请求：希望从 SM 片上 SRAM 中分给 SMEM 的百分比（0~100）
    //     -1 = cudaSharedmemCarveoutDefault（驱动决定）
    //     100 = cudaSharedmemCarveoutMaxShared（SMEM 最大，L1 最小）
    //     注：sm_86（Ampere）默认就已经是 100，驱动策略即为最大化 SMEM
    //     carveout 的实际效果是黑盒，CUDA 没有提供查询接口。这也是它被称为 preferred（偏好）而非 required（强制）的原因之一——驱动保留了忽略请求的权利，外部无法验证。



    // ── 编译期 vs 运行期 ───────────
    //
    //   编译期（ptxas 阶段）：
    //     静态分析 kernel 代码，确定 sharedSizeBytes / 寄存器数 / barrier 数
    //     写入 .cubin 元数据，不做任何实际分配
    //     示例输出：
    //       ptxas info: Used 31 registers, used 1 barriers, 8192 bytes smem, 400 bytes cmem[0]
    //       ptxas info: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    //     spill=0 说明寄存器够用，无溢出
    //
    //   运行期（kernel 启动时）：
    //     驱动读取编译期元数据 + cudaFuncSetAttribute 的设置
    //     检查 SM 上是否有足够资源，有则调度 block，无则 block 等待
    //     cudaFuncSetAttribute 是纯运行期概念，对编译结果零影响


    // cudaFuncAttributeMaxDynamicSharedMemorySize 和 cudaFuncAttributePreferredSharedMemoryCarveout
    // ── funcAttributes.cudaFuncAttributeMaxDynamicSharedMemorySize ──────────
    //
    //   作用：将 maxDynamicSharedSizeBytes 从默认 40 KiB 提升，突破 sharedMemPerBlock 限制
    //   性质：硬性申请，驱动必须保证；失败时返回 CUDA 错误（必须 cudaCheck 才能发现）
    //   上限约束：dynamic + static ≤ sharedMemPerBlockOptin
    //     动态申请上限 = sharedMemPerBlockOptin - sharedSizeBytes = 99 - 8 = 91 KiB
    //     sharedMemPerBlockOptin = 99 KiB（硬件 opt-in 上限，单 block 总 SMEM 天花板）
    //     sharedSizeBytes        =  8 KiB（静态 __shared__ 已占用）
    //     剩余可动态申请上限     = 91 KiB（reservedSharedMemPerBlock 已包含在 sharedMemPerBlockOptin 内）
    //   设置 96*1024：96 + 8 = 104 KiB > 99 KiB → 超出上限，CUDA 拒绝，返回错误
    //   设置 91*1024：91 + 8 =  99 KiB ≤ 99 KiB → 合法，maxDynamicSharedSizeBytes = 91 KiB（理论最大值）
    //   设置 90*1024：90 + 8 =  98 KiB ≤ 99 KiB → 合法，maxDynamicSharedSizeBytes = 90 KiB（当前设置，略保守）
    //   不调用此函数：maxDynamicSharedSizeBytes 维持默认 40 KiB，
    //                 <<<>>> 如果传入 >40 KiB 会触发 cudaErrorInvalidConfiguration
    //
    // ── funcAttributes.cudaFuncAttributePreferredSharedMemoryCarveout ───────
    //
    //   作用：纯粹的性能调优 hint，唯一实际影响的是 L1 cache 的物理大小
    //   性质：偏好提示（preferred），驱动可忽略；不影响任何容量上限

    //
    //   驱动的决策逻辑（三步）：
    //     1. 先保证 kernel 能运行（static + dynamic SMEM 需求是硬性约束，必须满足）
    //     2. 在此前提下，按 carveout 偏好分配剩余空间给 L1
    //     3. carveout 是 preferred，驱动可因其他约束（如同 SM 上其他 kernel）而偏离
    //
    //   【两个维度独立：per-block 软件上限 vs SM 级物理分区】
    //
    //   第三参数（per-block，软件层）：每个 block 实际申请多少 dynamic SMEM
    //     默认上限 = deviceProp.sharedMemPerBlock - sharedSizeBytes = 48 - 8 = 40 KiB
    //     不设置第三参数（或传 0）：实际 dynamic = 0，每 block 只占 static 8 KiB
    //     设置 MaxDynamicSharedMemorySize 可将上限提升至 ~100 KiB
    //
    //   carveout（SM 级，物理层）：SM 上 SMEM/L1 分区边界定在哪里
    //     与每个 block 实际用多少无关，是整个 SM 的物理设置
    //     L1 是 SM 级资源，SM 上所有并发 block 和 warp 共享同一个 L1
    //
    //   SMEM 分区大小 = max(实际需求, carveout 偏好值)
    //     carveout=MaxShared：偏好值 = 100 KiB → 分区始终顶到 100 KiB
    //     carveout=MaxL1    ：偏好值 = 实际需求 → 分区压到最小
    //
    //   【关键：SMEM 分区内闲置的空间不会变成 L1】
    //
    //   例A（kernel_3，static=8 KiB，第三参数=0，2 block 并发）：
    //
    //     carveout=MaxShared：
    //       每 block SMEM = 8+0 = 8 KiB，实际需求 = 8×2 = 16 KiB
    //       SMEM 分区 = max(16, 100) = 100 KiB（84 KiB 闲置但不能给 L1）
    //       L1（SM 级）= 128 - 100 = 28 KiB
    //
    //     carveout=MaxL1：
    //       每 block SMEM = 8+0 = 8 KiB，实际需求 = 8×2 = 16 KiB
    //       SMEM 分区 = min(16, 100) = 16 KiB
    //       L1（SM 级）= 128 - 16 = 112 KiB
    //
    //   即使第三参数=0（每 block dynamic=0），carveout=MaxShared 下
    //   L1 依然只有 28 KiB——因为分区边界由 carveout 决定，不由实际用量决定
    //
    //   例B（static=8 KiB，第三参数=40 KiB，2 block 并发）：
    //
    //     carveout=MaxShared：
    //       每 block SMEM = 8+40 = 48 KiB，实际需求 = 48×2 = 96 KiB
    //       SMEM 分区 = max(96, 100) = 100 KiB（4 KiB 闲置）
    //       L1（SM 级）= 128 - 100 = 28 KiB
    //
    //     carveout=MaxL1：
    //       每 block SMEM = 8+40 = 48 KiB，实际需求 = 48×2 = 96 KiB
    //       SMEM 分区 = min(96, 100) = 96 KiB
    //       L1（SM 级）= 128 - 96 = 32 KiB
    //
    //   第三参数=40 KiB 时，两种 carveout 下 L1 差距缩小（28 vs 32 KiB）
    //   因为实际需求 96 KiB 已接近 100 KiB 上限，MaxL1 的压缩空间所剩无几
    //
    //   例C：【同时设置 MaxL1 carveout 和大 MaxDynamicSharedMemorySize 时驱动被迫偏离的例子】
    //
    //   代码：
    //     cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
    //                          cudaSharedmemCarveoutMaxL1);        // 偏好：L1 最大
    //     cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
    //                          48 * 1024);                         // 允许 48 KiB dynamic
    //     kernel<<<grid, block, 40 * 1024>>>(...)                  // 实际传入 40 KiB
    //
    //   sm_86 上的计算（total SRAM = 128 KiB，static SMEM = 8 KiB）：
    //     每 block SMEM = static 8 KiB + dynamic 40 KiB = 48 KiB
    //     并发 block 数 = floor(100 KiB / 48 KiB) = 2 block       ← SMEM 约束推导
    //     实际需求      = 48 KiB × 2 block = 96 KiB
    //     SMEM 分区     = 96 KiB（MaxL1 将分区压到最小，最小即实际需求 96 KiB）
    //                   = min(实际需求 96, 硬件上限 100) = 96 KiB
    //     L1（SM 级）   = 128 - 96 = 32 KiB                       ← 被迫结果
    //
    //   carveout=MaxL1 偏好 L1≈112 KiB（实际需求仅 16 KiB 时可实现，见例A）
    //   但此处实际需求已达 96 KiB，SMEM 分区无法低于 96 KiB
    //   → L1 实际只有 32 KiB，而非 MaxL1 期望的 112 KiB
    //   → carveout 是 hint（偏好），驱动优先满足 SMEM 硬性需求，偏好只在剩余空间内生效

    //   物理分区示意（sm_86 总 SRAM = 128 KiB）：
    //     carveout = MaxShared (100%)：SMEM ≈ 100 KiB，L1 ≈  28 KiB（最小）
    //     carveout = MaxL1    (  0%)：SMEM = 满足 kernel 实际需求（最小，如 8 KiB），
    //                                 L1 ≈ 120 KiB（剩余尽可能多）
    //     → 核心逻辑："在满足 SMEM 需求的前提下，把剩余空间尽量给 L1"
    //
    //   carveout 影响什么（性能层面）：
    //     L1 越大 → global memory 访问命中率越高 → 依赖 L1 自动缓存的 kernel 受益
    //     SMEM 分区越大 → SM 上并发 block 的 SMEM 物理容量越充足 → occupancy 越高
    //
    //   carveout 不影响什么：
    //     maxDynamicSharedSizeBytes（软件配额，只有 MaxDynamicSharedMemorySize 能改）
    //     sharedMemPerBlock（历史软件默认值，与物理分区完全无关）
    //     kernel 能否 launch（由 maxDynamicSharedSizeBytes 决定，不由 carveout 决定）
    //     → 实验验证：carveout=MaxL1 时 maxDynamicSharedSizeBytes 依然是 90 KiB，kernel 正常运行
    //
    //   常见误解：当 carveout=MaxShared 且 maxDynamic=40 KiB 时，
    //     误以为 L1 = sharedMemPerMultiprocessor(100) - maxDynamic(40) - static(8) = 52 KiB
    //     （即把软件字段当物理空间来扣减）
    //
    //     错误原因一：maxDynamicSharedSizeBytes 是软件申请上限，不是物理占用量
    //       kernel_3 启动时第三个参数缺省（=0），实际 dynamic SMEM 物理占用 = 0，不是 40 KiB
    //       物理占用（per block）= static(8 KiB) + dynamic实际传入(0 KiB) = 8 KiB
    //
    //     错误原因二：L1/SMEM 物理分区的决定机制
    //       sharedMemPerMultiprocessor = 100 KiB 同样是硬件上限，不是实际分配值
    //
    //       实际分区由两步决定：
    //         第一步：确定 SMEM 的"地板"（硬性约束，必须满足，与 carveout 无关）
    //           地板 = (sharedSizeBytes + dynamic实际传入) × SM上并发block数
    //           kernel_3：(8 KiB + 0 KiB) × 2 块 = 16 KiB
    //           注意：dynamic"实际传入"= <<<>>> 第三个参数值，不是 maxDynamicSharedSizeBytes
    //                 maxDynamicSharedSizeBytes=40 KiB 是软件上限，不影响物理地板
    //
    //         第二步：carveout 决定"地板以上的剩余空间"如何分配
    //           carveout=MaxShared → 剩余空间尽量给 SMEM，L1 取最小值
    //           carveout=MaxL1    → 剩余空间尽量给 L1，SMEM 只保留地板
    //
    //       只能推算范围（精确值无法通过 CUDA API 读取，需 ncu profiler 观测）：
    //         SM 总 SRAM         = 128 KiB（硬件固定）
    //         SMEM 物理分区      ≤ 100 KiB（carveout=MaxShared 时趋近上限）
    //         L1  物理分区       ≥  28 KiB（= 128 - 100，最小保留量）
    //
    //   sm_86 特殊情况：
    //     Ampere 驱动默认 carveout 已是 100，初始值就是 100（非其他架构的 -1）
    //     设置 MaxShared 是空操作；设置 MaxL1 理论上减少 L1，但对本 kernel 无影响
    //
    //   为什么 kernel_3 设置 carveout 没有可观测效果：
    //     kernel_3 所有 global memory 数据都经由 __shared__ 手动管理，
    //     计算热循环完全不走 L1 自动缓存，L1 大小对结果和性能无影响
    //     原始注释："This doesn't currently make a difference,
    //                since occupancy is limited by reg and thread count"
    //     保留此调用是跨架构最佳实践：Pascal/Volta 默认 carveout 非 100，
    //     不设置则物理 SMEM 分区可能不足，影响 occupancy
    //
    // ── 两个属性的关系与协作 ────────
    //
    //   MaxDynamicSharedMemorySize（软件层面，per-kernel 申请配额）：
    //     类比：向 OS 申请 ulimit，决定"kernel 被允许申请多少"
    //     不设置：最多 40 KiB dynamic，超过则 kernel launch 失败
    //     设置后：最多 90 KiB dynamic，可在 <<<>>> 第三个参数传入大 SMEM
    //
    //   PreferredSharedMemoryCarveout（硬件层面，SM 资源分区偏好）：
    //     类比：告诉 OS "我偏好大页内存"，决定"SM 如何划分物理 SRAM"
    //     不设置（sm_86 默认已 100）：SM 物理 SRAM 已尽量分给 SMEM
    //     设置后：同上，在其他架构（Pascal/Volta）上才有实质意义
    //
    //   两个属性的一般性原则（适用于真正使用大量 dynamic SMEM 的 kernel）：
    //     只设 MaxDynamic=90 KiB，不设 carveout
    //       → kernel 被允许申请 90 KiB，但在 Pascal/Volta 等默认 carveout 非 100% 的架构上
    //         物理 SMEM 分区可能不足，导致 SM 上能并发的 block 数减少（occupancy 下降）
    //         注意：carveout 是 preferred hint，驱动在 launch 时会按需调整，通常不会导致 block 无法调度
    //     只设 carveout=100，不设 MaxDynamic
    //       → SM 物理分配偏 SMEM，但 kernel 仍受 40 KiB 软件上限，<<<>>> 无法传入更大值
    //         多余的物理 SMEM 被浪费，kernel 实际无法使用超过 40 KiB 的动态 SMEM
    //     两者同时设置：
    //       → kernel 被允许申请 90 KiB，SM 也倾向分配足够物理 SMEM，occupancy 最优
    //
    //   对 kernel_3 的实际情况：
    //     kernel_3 只用 8 KiB 静态 SMEM，<<<>>> 第三个参数缺省 = 0，实际动态 SMEM 用量 = 0
    //     MaxDynamic=90 KiB 设了但从未实际使用，两个属性在本机 sm_86 均无可观测效果
    //     原始注释："This doesn't currently make a difference"——这才是最准确的描述
    //     保留这两个调用是跨架构最佳实践，而非 kernel_3 本身的功能需要


    // ── cudaFuncSetAttribute 的持久性 ──────────
    //
    //   cudaFuncSetAttribute 修改的是 CUDA context 内 per-kernel 的全局属性，
    //   效果在同一进程内永久有效，不会因为函数返回而重置
    //   因此 benchmark 循环从第二轮起，"before" 读取的已是上一轮设置过的值
    //   → 想观察真正的"设置前默认值"，必须只在第一次调用时写文件（record 机制）

    // ── 本 kernel 在 sm_86 上的实验结果总结 ───────────
    //
    //   真实默认值（进程启动后第一次读取）：
    //     maxDynamicSharedSizeBytes = 40 KiB（= 48 - 8），preferredShmemCarveout = 100
    //     注：sm_86 carveout 默认 100，不是 -1（其他架构默认 -1 = cudaSharedmemCarveoutDefault）
    //
    //   设置 MaxDynamic=90*1024 后：
    //     maxDynamicSharedSizeBytes = 90 KiB ✓
    //
    //   设置 carveout=MaxShared 后（sm_86 上等于空操作）：
    //     preferredShmemCarveout = 100（未变）
    // ══════════════════════════════════════════════════════════════════════════

    // cudaCheck(cudaFuncSetAttribute(gemm_shared_mem_block<32>,
    //                     cudaFuncAttributeMaxDynamicSharedMemorySize,
    //                     91*1024));

    // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
    // out all of L1 to SMEM. This doesn't currently make a difference, since
    // occupancy is limited by reg and thread count, but it's good to do anyway.
    // （sm_86 上 carveout 默认已是 100，此调用在本机等于空操作；
    //  保留此调用是为了在 Pascal/Volta 等默认 carveout 非 100 的架构上也能正确运行）
    cudaCheck(cudaFuncSetAttribute(gemm_shared_mem_block<32>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared));


    cudaCheck(cudaGetDeviceProperties_v2(&deviceProp, deviceIdx));
    cudaCheck(cudaFuncGetAttributes(&funcAttributes, reinterpret_cast<const void*>(gemm_shared_mem_block<32>)));

    file << "after cudaFuncSetAttribute,\n";
    file << "device properties are listed as below: \n";
    file << "deviceProp.sharedMemPerMultiprocessor: ";
    // / 优先级高于 <<，(1<<10)=1024，括号防止 1<<10 被解析为流插入运算符
    file << deviceProp.sharedMemPerMultiprocessor / (1<<10);
    file << " KiB";
    file << "\n";
    file << "deviceProp.reservedSharedMemPerBlock: ";
    file << deviceProp.reservedSharedMemPerBlock / (1<<10);
    file << " KiB";
    file << "\n";
    file << "deviceProp.sharedMemPerBlock: ";
    file << deviceProp.sharedMemPerBlock / (1<<10);
    file << " KiB";
    file << "\n";
    file << "deviceProp.sharedMemPerBlockOptin: ";
    file << deviceProp.sharedMemPerBlockOptin  / (1<<10);
    file << " KiB";
    file << "\n";
    file << "deviceProp.regsPerBlock: ";
    file << deviceProp.regsPerBlock;
    file << "\n";
    file << "deviceProp.regsPerMultiprocessor: ";
    file << deviceProp.regsPerMultiprocessor;
    file << "\n\n";
    file << "kernel_3 properties are listed as below: \n";
    file << "funcAttributes.sharedSizeBytes: ";
    file << funcAttributes.sharedSizeBytes / (1<<10);
    file << " KiB";
    file << "\n";
    file << "funcAttributes.maxDynamicSharedSizeBytes: ";
    file << funcAttributes.maxDynamicSharedSizeBytes / (1<<10);
    file << " KiB";
    file << "\n";
    file << "funcAttributes.preferredShmemCarveout: ";
    file << funcAttributes.preferredShmemCarveout;
    file << "\n";
    file << "funcAttributes.localSizeBytes: ";
    file << funcAttributes.localSizeBytes;
    file << "\n";
    file << "funcAttributes.numRegs: ";
    file << funcAttributes.numRegs;
    file << "\n\n";
  }

  gemm_shared_mem_block_v2<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
  cudaCheck(cudaGetLastError());
}

void rungemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  // gridDim(CEIL_DIV(N,BN), CEIL_DIV(M,BM))：
  //   kernel_4 中 cCol=blockIdx.x，cRow=blockIdx.y（x→列，y→行）
  //   → gridDim.x = CEIL_DIV(N,BN) = 4096/64 = 64：列方向共 64 个 block
  //   → gridDim.y = CEIL_DIV(M,BM) = 4096/64 = 64：行方向共 64 个 block
  //   → 总 block 数 64×64 = 4096，覆盖 C 的全部 BM×BN tile
  //
  // blockDim((BM*BN)/TM)：
  //   C tile 共 BM×BN = 64×64 = 4096 个元素，每个线程计算 TM=8 个
  //   → 需要 4096/8 = 512 个线程/block
  //   同时满足加载约束：BM*BK = BN*BK = 64*8 = 512（assert 验证）
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  gemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
  cudaCheck(cudaGetLastError());
}

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
    cudaCheck(cudaGetLastError());
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
    cudaCheck(cudaGetLastError());
  }
}

void runSgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
    cudaCheck(cudaGetLastError());
  }
}

void runSgemmResolveBankConflicts(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
    cudaCheck(cudaGetLastError());
  }
}

void runSgemmResolveBankExtraCol(int M, int N, int K, float alpha, float *A,
                                 float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
    cudaCheck(cudaGetLastError());
  }
}

void runSgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  // A100
  // const uint K9_BK = 16;
  // const uint K9_TM = 4;
  // const uint K9_TN = 4;
  // const uint K9_BM = 64;
  // const uint K9_BN = 64;
  // A6000
  const uint K9_BK = 16;
  const uint K9_TM = 8;
  const uint K9_TN = 8;
  const uint K9_BM = 128;
  const uint K9_BN = 128;
  dim3 blockDim(K9_NUM_THREADS);

  static_assert(
      (K9_NUM_THREADS * 4) % K9_BK == 0,
      "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
      "during each iteraion)");
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BN == 0,
      "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of As "
      "during each iteration)");
  static_assert(
      K9_BN % (16 * K9_TN) == 0,
      "K9_BN must be a multiple of 16*K9_TN to avoid quantization effects");
  static_assert(
      K9_BM % (16 * K9_TM) == 0,
      "K9_BM must be a multiple of 16*K9_TM to avoid quantization effects");
  static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BM*K9_BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BN*K9_BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
  sgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
  cudaCheck(cudaGetLastError());
}

void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Settings for A100
  // const uint K10_NUM_THREADS = 128;
  // const uint K10_BN = 128;
  // const uint K10_BM = 64;
  // const uint K10_BK = 16;
  // const uint K10_WN = 64;
  // const uint K10_WM = 32;
  // const uint K10_WNITER = 1;
  // const uint K10_TN = 4;
  // const uint K10_TM = 4;
  // Settings for A6000
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 8;
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                0);
  constexpr uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // warpsubtile in warptile
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K10_BN % (16 * K10_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K10_BM % (16 * K10_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
  cudaCheck(cudaGetLastError());
}

void runSgemmDoubleBuffering(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  // Settings for A100
  // const uint K11_NUM_THREADS = 256;
  // const uint K11_BN = 128;
  // const uint K11_BM = 64;
  // const uint K11_BK = 16;
  // const uint K11_WN = 32;
  // const uint K11_WM = 32;
  // const uint K11_WNITER = 2;
  // const uint K11_TN = 4;
  // const uint K11_TM = 4;
  // Settings for A6000
  const uint K11_NUM_THREADS = 256;
  const uint K11_BN = 256;
  const uint K11_BM = 128;
  const uint K11_BK = 16;
  const uint K11_WN = 32;
  const uint K11_WM = 128;
  const uint K11_WNITER = 1;
  const uint K11_TN = 8;
  const uint K11_TM = 8;
  dim3 blockDim(K11_NUM_THREADS);

  constexpr uint NUM_WARPS = K11_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K11_BN % K11_WN == 0) and (K11_BM % K11_WM == 0));
  static_assert((K11_BN / K11_WN) * (K11_BM / K11_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K11_WM * K11_WN) % (WARPSIZE * K11_TM * K11_TN * K11_WNITER) ==
                0);
  constexpr uint K11_WMITER =
      (K11_WM * K11_WN) / (32 * K11_TM * K11_TN * K11_WNITER);
  // warpsubtile in warptile
  static_assert((K11_WM % K11_WMITER == 0) and (K11_WN % K11_WNITER == 0));

  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BK == 0,
                "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BN == 0,
                "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K11_BN % (16 * K11_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K11_BM % (16 * K11_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K11_BM * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K11_BN * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K11_BN), CEIL_DIV(M, K11_BM));
  sgemmDoubleBuffering<K11_BM, K11_BN, K11_BK, K11_WM, K11_WN, K11_WNITER,
                       K11_TM, K11_TN, K11_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
  cudaCheck(cudaGetLastError());
}

void runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                              float *B, float beta, float *C) {
  // Settings for A6000
  const uint K12_NUM_THREADS = 128;
  const uint K12_BN = 128;
  const uint K12_BM = 128;
  const uint K12_BK = 16;
  const uint K12_WN = 64;
  const uint K12_WM = 64;
  const uint K12_WNITER = 4;
  const uint K12_TN = 4;
  const uint K12_TM = 8;
  dim3 blockDim(K12_NUM_THREADS);

  constexpr uint NUM_WARPS = K12_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K12_BN % K12_WN == 0) and (K12_BM % K12_WM == 0));
  static_assert((K12_BN / K12_WN) * (K12_BM / K12_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K12_WM * K12_WN) % (WARPSIZE * K12_TM * K12_TN * K12_WNITER) ==
                0);
  constexpr uint K12_WMITER =
      (K12_WM * K12_WN) / (32 * K12_TM * K12_TN * K12_WNITER);
  // warpsubtile in warptile
  static_assert((K12_WM % K12_WMITER == 0) and (K12_WN % K12_WNITER == 0));

  static_assert((K12_NUM_THREADS * 4) % K12_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K12_NUM_THREADS * 4) % K12_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K12_BN % (16 * K12_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K12_BM % (16 * K12_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K12_BM * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K12_BN * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K12_BN), CEIL_DIV(M, K12_BM));
  // 【模板惰性实例化（Template Lazy Instantiation）与编译警告 #20054-D 的关系】
  //
  // 12_kernel_double_buffering.cuh 中使用了：
  //   static __shared__ cuda::barrier<...> frontBarrier;
  //   static __shared__ cuda::barrier<...> backBarrier;
  //
  // nvcc 对此发出警告 #20054-D：
  //   "static" is not allowed on a __shared__ variable declared in a function template
  //   （在函数模板中声明的 __shared__ 变量不允许使用 static）
  //
  // 但如果注释掉下面这行调用，警告就消失了。原因是 C++ 的模板惰性实例化：
  //   - 模板函数（如 runSgemmDoubleBuffering2<...>）在定义时不会被编译
  //   - 只有当代码中出现具体的调用（实例化点）时，编译器才会为该模板生成实际代码
  //   - 没有调用 → 没有实例化 → 编译器从不处理模板体内的语句 → 不触发任何警告
  //
  // 对比普通函数：普通函数定义即编译，无论是否被调用，警告都会出现
  // 模板函数：按需编译，注释掉唯一的调用点，等价于让整段模板代码从未存在
  runSgemmDoubleBuffering2<K12_BM, K12_BN, K12_BK, K12_WM, K12_WN, K12_WNITER,
                           K12_TM, K12_TN, K12_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
  cudaCheck(cudaGetLastError());
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle, int deviceIdx, bool record) {
  //参数默认值只需要在头文件的函数声明中写就可以了，因为默认参数是在编译阶段由调用点决定的
  switch (kernel_num) {
  case 0:
    runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_gemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_gemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_gemm_shared_mem_block(M, N, K, alpha, A, B, beta, C,deviceIdx,record);
    break;
  case 4:
    rungemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 6:
    runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
    break;
  case 7:
    runSgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
    break;
  case 8:
    runSgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
    break;
  case 9:
    runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
    break;
  case 10:
    runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 11:
    runSgemmDoubleBuffering(M, N, K, alpha, A, B, beta, C);
    break;
  case 12:
    runSgemmDoubleBuffering2(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    // throw：抛出异常，中断当前函数执行，沿调用栈向上传播，直到被 catch 捕获
    // std::invalid_argument：C++ 标准异常类（继承自 std::logic_error → std::exception）
    //   表示传入参数不合法，构造函数接受一个描述字符串，可通过 e.what() 获取
    // 本项目 main() 中没有 try-catch，异常未被捕获会触发 std::terminate() 终止程序
    // 实际上 main() 已经在入口处校验了 kernel_num 范围（0-12），正常流程不会走到这里
    // 此处作为防御性编程的最后一道保障，防止 run_kernel 被其他地方以非法参数调用
    throw std::invalid_argument("Unknown kernel number");
  }
}
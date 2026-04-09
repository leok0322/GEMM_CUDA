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

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

// cudaCheck 是项目自定义的错误检查函数，配合 gemm.cu 中的宏：
//   #define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))
// 每次 CUDA API 调用后自动传入源文件名和行号，方便定位出错位置
void cudaCheck(cudaError_t error, const char *file, int line) {
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
};

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
}

void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  sgemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
  // out all of L1 to SMEM. This doesn't currently make a difference, since
  // occupancy is limited by reg and thread count, but it's good to do anyway.
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
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
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
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
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_gemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
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
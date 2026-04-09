// #pragma once：头文件保护符，确保同一个头文件在同一编译单元（.cu/.cpp）中只被展开一次
// 若没有它，同一文件被多次 #include 会导致重复定义错误（redefinition error）
// 例如：gemm.cu -> kernels.cuh -> 1_naive.cuh
//                -> runner.cuh  -> 1_naive.cuh  <- 第二次 #include，pragma once 阻止再次展开
// 等价的传统写法（include guard）：
//   #ifndef NAIVE_CUH
//   #define NAIVE_CUH
//   ...
//   #endif
// #pragma once 更简洁，且避免了宏名冲突，是现代 C/C++ 的主流用法
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
Matrix sizes:
MxK * KxN = MxN

  M : A 的行数，C 的行数
  N : B 的列数，C 的列数
  K : A 的列数 = B 的行数，即矩阵乘法的内维度（inner dimension）
      点积长度：tmp = Σ A[x][i] * B[i][y], i=0..K-1
      M、N 决定输出矩阵 C 的形状，K 决定每个元素需要累加多少次
*/

__global__ void gemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // 将 block 坐标 + block 内线程坐标 映射为 C 矩阵的全局行列坐标
  // 线程 (threadIdx.x, threadIdx.y) 即 block 内 linear_id = threadIdx.x + threadIdx.y*blockDim.x 的线程
  // 负责计算 C[x][y] = Σ A[x][i] * B[i][y], i=0..K-1
  //
  // 例：block(1,0), thread(5,3), blockDim(32,32)
  //   x = 1*32 + 5 = 37  → C 矩阵第 37 行
  //   y = 0*32 + 3 = 3   → C 矩阵第  3 列
  //   → 该线程计算 C[37][3]
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // 边界检查：grid 按 tile 对齐分配线程，实际矩阵尺寸不一定是 blockDim 的整数倍
  // 超出矩阵范围的线程直接退出，不做计算
  if (x < M && y < N) {
    // 每个线程负责计算 C[x][y] 一个元素
    // tmp = A 第 x 行 点乘 B 第 y 列 = Σ A[x][i] * B[i][y], i=0..K-1
    // 内存行主序展开：A[x][i] = A[x*K+i]，B[i][y] = B[i*N+y]

    // 【naive kernel 的访存模式分析】（以 warp 0 为例，线程 x=0..31，y=0）
    // 同一 warp 内线程 (x=0,y=0), (x=1,y=0), ..., (x=31,y=0)，在循环第 i 次时：
    //
    // 访问 A：
    // thread(x=0)  → A[0*K + i] = A[i]
    // thread(x=1)  → A[1*K + i] = A[K+i]
    // thread(x=2)  → A[2*K + i] = A[2K+i]
    // ...
    // 相邻线程地址间距 = K * sizeof(float)  ← 跨度为整行，不连续
    //
    // 访问 C：
    // thread(x=0)  → C[0*N + 0] = C[0]
    // thread(x=1)  → C[1*N + 0] = C[N]
    // thread(x=2)  → C[2*N + 0] = C[2N]
    // ...
    // 相邻线程地址间距 = N * sizeof(float)  ← 同样不连续
    //
    // 访问 B 更糟：
    // warp 内所有线程 y=0，访问 B[i*N + 0]  ← 32 个线程访问同一个地址

    // A[x*K+i]：同一 warp 内相邻线程地址间距 = K*sizeof(float)
    //   → 散乱访问，退化为 32 次独立内存事务，带宽利用率极低（最差）
    //
    // B[i*N+y]：warp 内所有线程 y 相同（=0），访问同一地址 B[i*N+0]
    //   → GPU 检测到地址相同，只发起 1 次内存读取，结果广播给 32 个线程
    //   → 广播（broadcast），效率等价于 coalesced access（碰巧最优）
    //
    // C[x*N+y]：同一 warp 内相邻线程地址间距 = N*sizeof(float)
    //   → 散乱访问，退化为 32 次独立内存事务（最差）
    //
    // 性能瓶颈在 A 和 C 的散乱访问，这也是此 kernel 称为 naive 的原因
    // kernel 2（global_mem_coalesce）通过调整线程与元素的映射关系解决此问题

    // 【为什么不用一个线程处理一个乘积，存入 shared memory 再并行求和？】
    // 方案：K 个线程各算一个乘积 A[x][i]*B[i][y]，再规约求和
    // 问题1 - 规约需要 block 内同步（__syncthreads()），跨 block 无法同步：
    //   K 可能达 4096，远超单 block 线程上限（1024），无法用一个 block 完成规约
    // 问题2 - 改用 atomicAdd 跨线程累加：
    //   atomicAdd(&C[x*N+y], A[x*K+i]*B[i*N+y])
    //   K 个线程竞争写同一地址，强制串行化，比单线程循环更慢
    // 正确优化方向（kernel 3 shared_mem_blocking）：
    //   不是拆分乘法，而是 block 内线程协作将 A/B 的 tile 加载到 shared memory
    //   多个线程复用同一份数据，减少 global memory 访问次数
    //   并行的瓶颈在访存而非计算，优化目标是降低 global memory 访问量
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // GEMM 标准公式：C = α*(A@B) + β*C
    //   缩放的意义：通过调整 alpha/beta 让一个函数覆盖多种运算，避免多次 kernel 调用
    //   alpha  beta   等价操作
    //   1.0    0.0    C = A×B          （普通矩阵乘）
    //   1.0    1.0    C = A×B + C      （矩阵乘后累加，如神经网络加偏置）
    //   0.5    0.0    C = 0.5 * A×B    （带缩放的矩阵乘）
    //
    //   性能意义：GPU 访存代价极高，合并"乘+缩放+叠加"进一次 kernel，
    //   比分步计算少读写两次 C 矩阵，显著减少 global memory 访问
    //   这也是 cuBLAS cublasSgemm 的参数语义
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
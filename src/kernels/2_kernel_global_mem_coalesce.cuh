#pragma once

#include <cassert>
// cassert：提供 assert() 宏，用于运行时断言
//   assert(condition)：条件为假时终止程序并打印文件名、行号、失败表达式
//   定义 NDEBUG 宏后（Release 构建）所有 assert 被预处理器删除，零运行时开销
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 【函数模板（Function Template）】
// template <const uint BLOCKSIZE>：声明一个编译期模板参数 BLOCKSIZE，类型为 const uint
//
// 模板本身不是函数，是"生成函数的配方"：
//   - 定义时编译器不生成任何代码，仅记录模板
//   - 只有当代码中出现具体调用（实例化点）时，编译器才为该参数值生成一份具体函数
//
// BLOCKSIZE 是非类型模板参数（non-type template parameter），不是类型参数（typename T 那种）
// 使用模板参数而非普通函数参数的原因：让值成为编译期常量
//
//   普通函数参数（运行时）：
//     __global__ void kernel(int BLOCKSIZE, ...) {
//         threadIdx.x / BLOCKSIZE   // 运行时除法，编译器不知道 BLOCKSIZE 是多少，无法优化
//     }
//
//   非类型模板参数（编译期）：
//     threadIdx.x / BLOCKSIZE   // BLOCKSIZE=32，编译器直接替换为 threadIdx.x / 32
//                               // 32 是 2 的幂 → 优化为 threadIdx.x >> 5（位移，更快）
//     threadIdx.x % BLOCKSIZE   // 同理 → threadIdx.x & 31（位与，更快）
//
//   / 和 % 是昂贵的整数运算，GPU 上代价更高；编译期常量使编译器能替换为等价位运算
//   还能做循环展开、寄存器分配优化等，普通参数做不到
//
// 【模板实例化（Template Instantiation）】
// 调用处：gemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(M, N, K, ...)
//   <32> 是模板实参，编译器将模板中所有 BLOCKSIZE 替换为字面量 32，
//   生成一份具体的 __global__ 函数，即：
//     __global__ void gemm_global_mem_coalesce(...)  // BLOCKSIZE 已固定为 32
//   若还有 <64> 的调用，编译器再生成另一份 BLOCKSIZE=64 的函数，两份代码独立存在
template <const uint BLOCKSIZE>
__global__ void gemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {


  // 【线程映射：把 warp 内的变化维度从"行"换成"列"】
  //
  // blockDim = (BLOCKSIZE*BLOCKSIZE, 1, 1)，1D 线程块，threadIdx.x = 0 ~ BLOCKSIZE²-1
  //
  // threadIdx.x / BLOCKSIZE：block 内行偏移（0 ~ BLOCKSIZE-1），每 BLOCKSIZE 个线程换一行
  // threadIdx.x % BLOCKSIZE：block 内列偏移（0 ~ BLOCKSIZE-1），在 0~BLOCKSIZE-1 循环
  //
  // warp 内 32 个线程（threadIdx.x = 32k ~ 32k+31，BLOCKSIZE=32）：
  //   cRow = blockIdx.x*32 + k         （全部相同，/ 结果相同）
  //   cCol = blockIdx.y*32 + 0,1,...,31 （连续递增，% 结果递增）
  //
  // 一个 block 内有 32 个 warp，覆盖 C 的一个 32×32 tile：
  //   warp 0：cRow=base+0，cCol=base_col ~ base_col+31
  //   warp 1：cRow=base+1，cCol=base_col ~ base_col+31
  //   ...（cRow 递增，cCol 范围相同）
  //
  // 【与 naive kernel 的访存对比（合并事务的核心）】
  // naive 的 warp 内：x（行）递增，y（列）相同
  // coalesced 的 warp 内：cRow（行）相同，cCol（列）递增
  //
  // 调换映射关系后，每次循环迭代 i 时：
  //
  //   访问 A[cRow*K + i]：
  //     naive     → 32个线程行不同，地址间距=K → 散乱，32次事务（最差）
  //     coalesced → 32个线程行相同，访问同一地址 → broadcast，1次事务 ✓
  //
  //   访问 B[i*N + cCol]：
  //     naive     → 32个线程列相同，访问同一地址 → broadcast，1次事务
  //     coalesced → 32个线程列连续，地址连续 → coalesced，1次事务 ✓
  //     （B 的访问两者都是 1 次事务，机制不同但代价相同）
  //
  //   访问 C[cRow*N + cCol]（写回）：
  //     naive     → 32个线程行不同，地址间距=N → 散乱，32次事务（最差）
  //     coalesced → 32个线程列连续，地址连续 → coalesced，1次事务 ✓
  //
  // 合并事务规则（warp 级别）：
  //   warp 内 32 个线程的地址落在同一 128字节对齐 cache line → 合并为 1 次事务
  //   32个float × 4字节 = 128字节，连续访问恰好填满 1 条 cache line
  //   地址散乱时退化为最多 32 次独立事务
  //
  //   合并针对的是单次内存事务（单次循环迭代），不是整个循环：
  //     每次迭代 i，warp 内 32 个线程各自发出 1 个地址，硬件判断这 32 个地址能否合并
  //     整个循环共 K 次迭代，每次独立判断，合并与否取决于该次迭代的地址分布
  //
  //   合并是 warp 内部的概念，每个 warp 独立发起自己的内存事务
  //   不同 warp 之间无需协调，各自判断是否能合并，互不影响
  //   多个 warp 若碰巧访问相同 cache line，由 L1/L2 缓存命中处理，与合并无关

  // 两个 kernel 的对比：
  //   ┌───────────────────┬─────────────────────────────────────┬─────────────────────────────────────┐
  //   │                   │                naive                │              coalesced              │
  //   ├───────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤
  //   │ A[x/cRow * K + i] │ 32个线程行不同，stride=K → 32次事务 │ 32个线程行相同 → broadcast，1次事务 │
  //   ├───────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤
  //   │ B[i*N + y/cCol]   │ 32个线程列相同 → broadcast，1次事务 │ 32个线程列连续 → coalesced，1次事务 │
  //   ├───────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤
  //   │ C写回             │ stride=N → 32次事务                 │ 列连续 → coalesced，1次事务         │
  //   └───────────────────┴─────────────────────────────────────┴─────────────────────────────────────┘
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  // 边界检查：gridDim 按 BLOCKSIZE 对齐分配，实际矩阵尺寸不一定是 BLOCKSIZE 的整数倍
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}
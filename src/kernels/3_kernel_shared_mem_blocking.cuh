#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 【kernel_3：Shared Memory Blocking】
//
// 核心思想：
//   将 global memory 的 A/B 矩阵分块（tile）加载到 shared memory，
//   block 内所有线程协作加载，加载完后每个线程从 shared memory 读取数据计算。
//   每个数据从 global memory 只读一次，在 shared memory 内被复用 BLOCKSIZE 次。
//
// 对比 kernel_1（naive）：
//   naive：每个线程独立从 global memory 读 K 次 A、K 次 B，无复用
//   kernel_3：block 内 BLOCKSIZE² 个线程协作，每个 tile 只读一次，复用 BLOCKSIZE 次
//
// 矩阵布局（行主序）：
//   A: M×K，A[row][col] = A[row*K + col]
//   B: K×N，B[row][col] = B[row*N + col]
//   C: M×N，C[row][col] = C[row*N + col]
//
// grid/block 划分：
//   gridDim  = (CEIL_DIV(M,BLOCKSIZE), CEIL_DIV(N,BLOCKSIZE))
//   blockDim = (BLOCKSIZE*BLOCKSIZE, 1, 1)  ← 一维线程排列
//   每个 block 负责计算 C 的一个 BLOCKSIZE×BLOCKSIZE 的 tile
//
// 算法时序（以 K=64, BLOCKSIZE=32 为例，共 2 次 tile 迭代）：
//
//   初始化：A/B/C 指针推进到本 block 负责区域起点，tmp=0
//
//   迭代 0（bkIdx=0）：
//     协作加载 A[cRow*BS][0:32]   → As   （global mem 读，coalesced）
//     协作加载 B[0:32][cCol*BS]   → Bs   （global mem 读，coalesced）
//     __syncthreads()                     （等待所有线程加载完成）
//     A += 32, B += 32*N                  （指针推进到下一个 tile）
//     内层点积 dotIdx=0..31：tmp += As[row][dotIdx] * Bs[dotIdx][col]
//     __syncthreads()                     （防止快线程覆盖慢线程还在读的 smem）
//
//   迭代 1（bkIdx=32）：
//     协作加载 A[cRow*BS][32:64]  → As
//     协作加载 B[32:64][cCol*BS]  → Bs
//     __syncthreads()
//     A += 32, B += 32*N
//     内层点积 dotIdx=0..31：tmp += As[row][dotIdx] * Bs[dotIdx][col]
//     __syncthreads()
//
//   写回：C[row][col] = alpha * tmp + beta * C[row][col]
//
// 两层循环的数学等价性：
//   令 k = bkIdx + dotIdx，外层枚举 tile（步长 BLOCKSIZE），内层枚举 tile 内偏移（步长 1）
//   两层合起来覆盖 k ∈ [0, K)，等价于完整的矩阵乘法求和
//   Σ(bkIdx) Σ(dotIdx) A[row][bkIdx+dotIdx] * B[bkIdx+dotIdx][col]
//   = Σ(k=0..K-1) A[row][k] * B[k][col] = C[row][col]

template <const int BLOCKSIZE>
__global__ void gemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {

  // ── 确定本 block 负责 C 矩阵的哪个 tile ────────────
  // gridDim.x = CEIL_DIV(M,32)（行方向），gridDim.y = CEIL_DIV(N,32)（列方向）
  // 本 block 负责 C[cRow*BS..(cRow+1)*BS-1][cCol*BS..(cCol+1)*BS-1]
  //
  // 【为什么 blockIdx.x 可以对应行，而 threadIdx.x 必须对应列？】
  //
  // threadIdx.x 必须对应列（N方向）——硬件强制：
  //   warp 内线程按 x 优先线性化：linear_id = threadIdx.x + threadIdx.y * blockDim.x
  //   同一 warp 内 threadIdx.x = 0..31 连续变化，threadIdx.y 不动
  //   内存行主序下，coalesced 访问要求 warp 内线程访问同一行连续地址
  //   → threadIdx.x 必须映射到列方向，否则访存不连续
  //
  // blockIdx.x 可以自由对应行（M方向）——无约束：
  //   block 之间没有 warp 这类硬件约束，blockIdx 排列顺序不影响任何 warp 内部的访存模式
  //   block(0,0) 和 block(1,0) 是独立执行单元，各自内部的 coalesced 由 threadIdx 决定
  //   blockIdx.x 赋予行还是列，对 coalesced 无任何影响，纯粹是命名惯例
  //   gridDim 只决定 block 的数量和范围，对 threadIdx 的排列没有任何影响
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // ── 静态 shared memory 分配 ──────────
  // As/Bs 各缓存一个 BLOCKSIZE×BLOCKSIZE 的 tile
  // BLOCKSIZE=32：每个数组 32×32×4 = 4096 bytes，合计 8192 bytes = 8 KiB
  // ptxas 输出：Used 31 registers, used 1 barriers, 8192 bytes smem, 400 bytes cmem[0]
  //   sharedSizeBytes = 8192（静态 __shared__ 变量，编译期确定）
  //
  // 【为什么不能用 {} 列表初始化？】
  // CUDA 语言规范禁止对 __shared__ 变量使用初始化器，编译器报错：
  //   error: dynamic initialization is not supported for __shared__ variables
  // 原因：__shared__ 是 on-chip SRAM，硬件没有"kernel 启动时自动清零"的机制
  //       普通栈变量分配时 CPU/GPU 可顺便置零；shared memory 必须由线程显式写入
  //       若需要初始化，只能在 kernel 内手动赋值后 __syncthreads()
  //
  // 【默认值是垃圾值吗？】
  // 是的，未初始化的 shared memory 含有不确定值（上一个 block 残留的数据）
  // 但此 kernel 不需要初始化，因为：
  //   每次外层迭代开始，协作加载会覆盖写入 As/Bs 的所有元素
  //   __syncthreads() 保证写完再读，点积计算读到的永远是本次加载的有效数据
  //   时序：写 As/Bs（覆盖垃圾值）→ __syncthreads() → 读 As/Bs（已是有效值）✓
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // ── 将一维线程 ID 映射为二维坐标 ────
  // blockDim = (BLOCKSIZE², 1, 1)，线程以一维排列
  // threadCol = threadIdx.x % BLOCKSIZE：tile 内的列坐标（0..BLOCKSIZE-1）
  // threadRow = threadIdx.x / BLOCKSIZE：tile 内的行坐标（0..BLOCKSIZE-1）
  // 同一 warp 内 32 个线程（threadIdx.x=0..31），BLOCKSIZE=32：
  //   threadRow 全部相同（=0），threadCol = 0,1,...,31（连续）
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // ── 将指针推进到本 block 负责区域的起始位置 ────────
  //
  // 【原始写法：指针推进 + 相对偏移】
  // 将 A/B/C 基地址推进到本 block 的起点，之后用固定的相对偏移索引
  // 每次外层迭代再推进 A += BLOCKSIZE, B += BLOCKSIZE*N，实现 tile 滑动
  //
  // A：本 block 负责 C 的第 cRow 行 tile → 需要 A 的第 cRow*BS 行起
  //   跨 cRow*BLOCKSIZE 行，每行 K 个元素 → A += cRow*BLOCKSIZE*K
  //   之后 A[threadRow*K + threadCol] 即 tile 内 (threadRow, threadCol) 的元素
  //
  // B：本 block 负责 C 的第 cCol 列 tile → 需要 B 的第 cCol*BS 列起
  //   在第 0 行内偏移 cCol*BLOCKSIZE 列（row=0 时不需乘 N）→ B += cCol*BLOCKSIZE
  //   之后 B[threadRow*N + threadCol] 即 tile 内 (threadRow, threadCol) 的元素
  //
  // C：输出到 (cRow, cCol) tile 的左上角
  //   → C += cRow*BLOCKSIZE*N + cCol*BLOCKSIZE
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  // ── 主循环：沿 K 方向逐 tile 累加 ───
  // 每个线程有自己的 tmp（寄存器变量），block 内 BLOCKSIZE² 个 tmp 互相独立
  // 每个 tmp 对应 C tile 内一个元素的完整点积累加器
  float tmp {0.0f};
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {

    // ── 协作加载：每线程加载 As/Bs 各一个元素 ──────
    // 一对一映射：thread(threadRow, threadCol) 加载 tile 内 (threadRow, threadCol) 的元素
    //
    // 加载 A（coalesced）：
    //   同一 warp 内 threadRow 相同，threadCol=0..31 连续
    //   → A[threadRow*K + 0..31]：同一行连续 32 个元素 → coalesced ✓
    //
    // 加载 B（coalesced）：
    //   同一 warp 内 threadRow 相同，threadCol=0..31 连续
    //   → B[threadRow*N + 0..31]：同一行连续 32 个元素 → coalesced ✓
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // ── 同步屏障1：等待 block 内所有线程完成加载 ───
    // 计算阶段需要读 As/Bs，必须等所有线程写完才能读，否则读到脏数据
    __syncthreads();

    // ── 推进指针到下一个 k_tile ──────
    // A 沿列方向（K方向）推进 BLOCKSIZE 列：+= BLOCKSIZE
    // B 沿行方向（K方向）推进 BLOCKSIZE 行：+= BLOCKSIZE * N
    // 步长 = BLOCKSIZE 是 tile 的宽度，与 warp size 数值相同但含义不同
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // ── 点积计算：从 shared memory 读取数据累加 ─────
    // As 访问：warp 内所有线程 threadRow 相同，dotIdx 相同 → 同一地址 → 广播，无 bank conflict
    // Bs 访问：warp 内 threadCol=0..31，dotIdx 相同 → 访问同行 32 个不同 bank → 无 bank conflict
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }

    // ── 同步屏障2：防止快线程覆盖慢线程还在读的 shared memory ────────────
    // 若无此屏障：快线程进入下一轮覆盖 As/Bs，慢线程还在读当前轮 → 数据竞争
    __syncthreads();
  }

  // ── 写回结果：C = α*(A@B) + β*C ─────
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}


// ════════════════════════════════════════════════════════════════════════════
// 【复写版本对比分析】
//
// 与原始写法的区别：绝对索引 vs 指针推进
//
// 原始写法（上方）：
//   先将 A/B/C 指针推进到本 block 起点，之后用相对偏移索引
//   每次迭代 A += BLOCKSIZE, B += BLOCKSIZE*N 实现滑动
//   优点：[] 内表达式简洁，只需局部坐标
//
// 复写写法（下方）：
//   A/B/C 指针不动，用 tileIdx 显式计算每次迭代的全局坐标
//   A[totalRow*K + tileIdx + col]，B[(tileIdx+row)*N + totalCol]
//   优点：每行索引含义一目了然，不依赖指针状态
//
// 两种写法数学等价，访存模式相同（coalesced），性能相同：
//   相同的 shared memory 用量（8 KiB）
//   相同的 global memory 访问次数
//   相同的 bank conflict 分析结果
//   编译器对两种写法均可生成相同的 PTX 指令
//
// 复写版本的已知 Bug（供学习参考）：
//   Bug1：blockInitCol = blockIdx.x * BLOCKSIZE → x/y 方向与原始相反
//         应为：blockInitRow = blockIdx.x * BLOCKSIZE（x→行，y→列）
//   Bug2：if (totalCol < M && totalRow < N) → M/N 搞反
//         应为：if (totalRow < M && totalCol < N)
//   Bug3：复写代码位于同一 kernel 函数内，两段代码均会执行
//         应将复写版本提取为独立 kernel 函数
// ════════════════════════════════════════════════════════════════════════════

template <const int BLOCKSIZE>
__global__ void gemm_shared_mem_block_v2(int M, int N, int K, float alpha,
                                         const float *A, const float *B,
                                         float beta, float *C) {

  // blockIdx.x → 行方向（M），blockIdx.y → 列方向（N）
  int blockInitRow = blockIdx.x * BLOCKSIZE;   // 本 block 在 C 中的起始行
  int blockInitCol = blockIdx.y * BLOCKSIZE;   // 本 block 在 C 中的起始列

  // 线程局部坐标（tile 内）
  int row = threadIdx.x / BLOCKSIZE;           // tile 内行坐标（0..BS-1）
  int col = threadIdx.x % BLOCKSIZE;           // tile 内列坐标（0..BS-1）

  // 全局坐标（C 矩阵中的绝对位置）
  int totalRow = blockInitRow + row;           // C 的行
  int totalCol = blockInitCol + col;           // C 的列

  __shared__ float Ashared[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bshared[BLOCKSIZE * BLOCKSIZE];

  float temp {0.0f};
  if (totalRow < M && totalCol < N) {
    for (int tileIdx = 0; tileIdx < K; tileIdx += BLOCKSIZE) {

      // A：行坐标 totalRow（M方向），列坐标 tileIdx+col（K方向滑动）
      // coalesced：warp 内 col=0..31 连续 → A 同一行连续 32 元素
      Ashared[row * BLOCKSIZE + col] = A[totalRow * K + (tileIdx + col)];

      // B：行坐标 tileIdx+row（K方向滑动），列坐标 totalCol（N方向）
      // coalesced：warp 内 col=0..31 连续 → totalCol 连续 → B 同一行连续 32 元素
      Bshared[row * BLOCKSIZE + col] = B[(tileIdx + row) * N + totalCol];

      __syncthreads();

      // 点积：As 广播（无 bank conflict），Bs 连续访问（无 bank conflict）
      for (int i = 0; i < BLOCKSIZE; ++i) {
        temp += Ashared[row * BLOCKSIZE + i] * Bshared[i * BLOCKSIZE + col];
      }

      __syncthreads();
    }
  }

  // 写回（coalesced）：warp 内 col 连续 → C 同一行连续 32 元素
  C[totalRow * N + totalCol] = alpha * temp + beta * C[totalRow * N + totalCol];
}

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ══════════════════════════════════════════════════════════════════════════════
// kernel_5：2D Blocktiling
//
// ── 每线程寄存器拆解（BM=128, BN=128, BK=8, TM=8, TN=8）────────────────────
//
// 【显式寄存器数组（最大头）】
//   threadResults[TM×TN] = 8×8 = 64 个   ← 2D tiling 的累加器，比 kernel_4 多 TN 倍
//   regM[TM]             = 8 个           ← 每 dotIdx 从 As 加载 TM 个元素缓存到寄存器
//   regN[TN]             = 8 个           ← 每 dotIdx 从 Bs 加载 TN 个元素缓存到寄存器
//   小计：80 个
//
// 【标量变量】
//   cRow, cCol                             : 2   block 坐标
//   threadCol, threadRow                   : 2   计算阶段线程坐标
//   innerRowA, innerColA, strideA          : 3   As 加载坐标
//   innerRowB, innerColB, strideB          : 3   Bs 加载坐标
//   bkIdx, dotIdx, loadOffset×2, i×2,
//   resIdxM, resIdxN                       : 8   循环计数器
//   A, B, C 指针（64-bit，各占 2 个寄存器）: 6
//   alpha, beta                            : 2
//   地址计算中间量                          : ~15
//   小计：~41 个
//
// 合计：约 121 个寄存器/线程
//
// ── 与 kernel_4 的对比 ────────────────────────────────────────────────────
//
//                  kernel_4          kernel_5
//   显式数组     threadResults[8]    threadResults[64] + regM[8] + regN[8]
//                + tmpB = 9 个       = 80 个
//   合计         ~48 个              ~121 个
//   增加原因     —                   TN 方向也展开，累加器从 TM 变为 TM×TN
//
// ── 每 block 寄存器与 occupancy ───────────────────────────────────────────
//
//   blockDim.x = BM*BN/(TM*TN) = 128*128/64 = 256 线程/block
//   每 block 寄存器 = 256 × 121 ≈ 30976 个
//   SM 总寄存器     = 65536 个
//   每 SM 最多驻留  = 65536 / 30976 ≈ 2 个 block（寄存器是 occupancy 瓶颈）
//
// ── spill 的触发条件 ────────────────────────────────────────────────────────
//
// spill = 单线程寄存器需求 > 编译器给该线程设定的上限
// 上限来源（优先级从高到低）：
//
//   来源                          每线程上限
//   硬件绝对上限                  255 个（无论如何不能超过）
//   __launch_bounds__(256, 1)    65536 / 256       = 256 个
//   __launch_bounds__(256, 2)    65536 / (256×2)   = 128 个（保证 2 block 同时驻留）
//   --maxrregcount=32            32 个（nvcc 编译选项硬性上限）
//   无任何声明（默认）            65536 / 1024      = 64 个（按最坏情况 1024 线程估算）
//
// 典型触发场景：
//   场景1：大寄存器数组 + 无 __launch_bounds__
//     threadResults[64] = 64 个，上限恰好 64 → regM[8]/regN[8] 无处安放 → spill
//   场景2：minBlocksPerSM 设太大，压低了上限
//     __launch_bounds__(256, 4) → 上限 = 65536/(256×4) = 64 → 同上
//   场景3：kernel 本身变量超过 255 个
//     硬件绝对上限，必然 spill，无法避免
//
// spill 的代价：
//   ptxas 将 spill 变量放入 local memory（物理上是 HBM），生成 ld.local/st.local 指令
//   寄存器访问 ~1 cycle，local memory ~300 cycles（L1 未命中时）
//   ptxas 编译输出可见：
//     "N bytes spill stores, N bytes spill loads"  ← 有值说明有 spill
//     "0 bytes spill stores, 0 bytes spill loads"  ← 无 spill ✓
//
// blockDim 过大不会触发 spill：
//   GPU 寄存器是编译期静态预分配，运行时不会动态溢出
//   若 numRegs × blockDim.x > regsPerBlock(65536) → kernel 启动失败（不是 spill）
//   若满足条件 → 所有线程寄存器预分配完毕，执行期数量固定不变
//
// ── __launch_bounds__((BM*BN)/(TM*TN), 1) 的作用 ──────────────────────────
//
//   不写时：编译器按最坏情况（1024 线程/block）估算每线程可用寄存器上限
//     65536 / 1024 = 64 个/线程
//     threadResults[64] 本身就占满 64 个 → regM[8] 和 regN[8] 必然 spill
//     spill 到 local memory（实为 HBM），每次访问约 300 cycles，严重拖慢性能
//
//   写了之后：编译器知道 blockDim.x 最大为 256
//     65536 / 256 = 256 个/线程 ← 充裕
//     threadResults、regM、regN 全部驻寄存器，无 spill ✓
//     第二个参数 minBlocksPerSM=1：不要求高 occupancy，优先保证寄存器够用
//     （若写 2，编译器会限制寄存器数以装下 2 个 block，可能重新触发 spill）
// ══════════════════════════════════════════════════════════════════════════════
template <const int BM, const int BN, const int BK, const int TM, const int TN>
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
//   maxThreadsPerBlock = (BM*BN)/(TM*TN) = 256：告知编译器 blockDim.x 的实际上限
//     编译器据此推算每线程可用寄存器上限 = SM总寄存器 / maxThreadsPerBlock
//     = 65536 / 256 = 256 个（而非按默认 1024 线程估算的 64 个）
//   minBlocksPerSM = 1：每 SM 至少驻留 1 个 block，不限制寄存器用量，优先避免 spill
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    gemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
  gemm2DBlocktiling_v2(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C) {

  // 每个block的起始行和列
  const uint InitRow {blockIdx.y * BM};
  const uint InitCol {blockIdx.x * BN};

  // ── 线程 → 输出位置映射（行优先线性化，与 1D blocktiling 本质相同）──────────────
  //
  // 1D blocktiling（kernel_4）：每线程负责 1 列，BN 个列组
  //   assert(blockDim.x % BN == 0);
  //   threadCol = threadIdx.x % BN;      // 第几列（1 列/线程）
  //   threadRow = threadIdx.x / BN;      // 第几行组
  //
  // 2D blocktiling（本 kernel）：每线程负责 TN 列，BN/TN 个列组
  //   assert(blockDim.x % (BN/TN) == 0);
  //   threadColGroup = threadIdx.x % (BN/TN);  // 第几列组（TN 列/线程）
  //   threadRowGroup = threadIdx.x / (BN/TN);  // 第几行组
  //
  // 公式形式完全相同：% X → 列（组）编号，/ X → 行组编号
  // 唯一区别是粒度：X = BN（1列）vs X = BN/TN（TN列）
  // assert 目的也相同：保证 blockDim.x 能被 X 整除，行组不出现分数

  // 保证一个block能处理完整的n行组，BN是总列数，TN是每个线程要处理的列数
  // 它保证的是：线程能完整覆盖输出 block tile 的所有行组。
  assert(blockDim.x % (BN / TN) == 0);
  // 当前线程负责的行组和列组
  const uint threadRowGroup{threadIdx.x / (BN / TN)};
  const uint threadColGroup{threadIdx.x % (BN / TN)};


  // SMEM静态分配
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];


  // ══════════════════════════════════════════════════════════════════════════════
  // 四个必要约束及违反后果（附完整数值反例）
  //
  // ① assert(blockDim.x % BK == 0)
  //   作用：保证 strideA = blockDim.x/BK 是整数（每轮推进整数行）
  //
  //   反例：BK=3, blockDim.x=8, BM=8
  //   BM*BK=24, 24%8=0 → smemNum=3（整数，assert②成立），8%3=2≠0（assert①违反）
  //   线程映射（innerRowGroupAs=threadIdx.x/3, innerColAs=threadIdx.x%3）：
  //     thread 0,1,2 → innerRowGroupAs=0, innerColAs=0,1,2
  //     thread 3,4,5 → innerRowGroupAs=1, innerColAs=0,1,2
  //     thread 6,7   → innerRowGroupAs=2, innerColAs=0,1  ← col2 无线程（需threadIdx.x=8，不存在）
  //   As 共 24 个元素（row0~7，col0~2）
  //
  //   floor strideA=2，写入地址 As[(innerRowGroupAs+rowIdx*2)*3+innerColAs]：
  //     rowIdx=0: thread0→As[0]=row0col0  thread1→As[1]=row0col1  thread2→As[2]=row0col2
  //               thread3→As[3]=row1col0  thread4→As[4]=row1col1  thread5→As[5]=row1col2
  //               thread6→As[6]=row2col0  thread7→As[7]=row2col1
  //     rowIdx=1: thread0→As[6]=row2col0(重写) thread1→As[7]=row2col1(重写) thread2→As[8]=row2col2
  //               thread3→As[9]=row3col0  thread4→As[10]=row3col1 thread5→As[11]=row3col2
  //               thread6→As[12]=row4col0 thread7→As[13]=row4col1
  //     rowIdx=2: thread0→As[12]=row4col0(重写) thread1→As[13]=row4col1(重写) thread2→As[14]=row4col2
  //               thread3→As[15]=row5col0 thread4→As[16]=row5col1 thread5→As[17]=row5col2
  //               thread6→As[18]=row6col0 thread7→As[19]=row6col1
  //   结果：row0~5 完整 ✓，row6 仅 col0,col1（As[20]=row6col2 未写）✗，row7 完全未写（As[21..23]）✗
  //
  //   ceil strideA=3，写入地址 As[(innerRowGroupAs+rowIdx*3)*3+innerColAs]：
  //     rowIdx=0: thread0→As[0]  ...  thread5→As[5]  thread6→As[6]=row2col0  thread7→As[7]=row2col1
  //     rowIdx=1: thread0→As[9]=row3col0 ... thread5→As[14]=row4col2
  //               thread6→As[15]=row5col0  thread7→As[16]=row5col1
  //     rowIdx=2: thread0→As[18]=row6col0 ... thread5→As[23]=row7col2 ✓
  //               thread6→As[24] 越界  thread7→As[25] 越界（As 共 24 个元素，索引 0~23）✗
  //   永远未写：As[8]=row2col2，As[17]=row5col2 → 结果错误 ✗
  //
  // ② assert(BM * BK % blockDim.x == 0)
  //   作用：保证 smemNum = BM×BK/blockDim.x 是整数（迭代次数精确）
  //   前提：① 已成立（strideA 是整数）
  //
  //   反例：BM=9, BK=4, blockDim.x=8
  //   strideA=8/4=2（整数 ✓），smemNum_exact=36/8=4.5
  //   As 共 36 个元素（row0~8），innerRowGroupAs：thread0-3→0，thread4-7→1
  //
  //   floor smemNum=4：
  //     rowIdx=0: thread0-3→As[0..3]=row0   thread4-7→As[4..7]=row1
  //     rowIdx=1: thread0-3→As[8..11]=row2  thread4-7→As[12..15]=row3
  //     rowIdx=2: thread0-3→As[16..19]=row4 thread4-7→As[20..23]=row5
  //     rowIdx=3: thread0-3→As[24..27]=row6 thread4-7→As[28..31]=row7
  //     row8（As[32..35]）永远未写 → 结果错误 ✗
  //
  //   ceil smemNum=5：
  //     rowIdx=0~3: 同上，row0~7 正确写入
  //     rowIdx=4: thread0-3→As[32..35]=row8 ✓
  //               thread4-7→As[36..39] ← 越界（As 共 36 个元素，索引 0~35）✗
  //   row8 正确加载，但 thread4-7 写 As[36..39]→ SMEM 溢出 ✗
  //
  // ③ assert(blockDim.x % BN == 0)
  //   作用：保证 strideB = blockDim.x/BN 是整数，与 ① 完全对称，对象换为 Bs
  //
  //   反例：BN=3, blockDim.x=8, BK=8
  //   BK*BN=24, 24%8=0 → smemNum=3（整数，assert④成立），8%3=2≠0（assert③违反）
  //   线程映射（innerRowGroupBs=threadIdx.x/3, innerColBs=threadIdx.x%3）：
  //     thread 0,1,2 → innerRowGroupBs=0, innerColBs=0,1,2
  //     thread 3,4,5 → innerRowGroupBs=1, innerColBs=0,1,2
  //     thread 6,7   → innerRowGroupBs=2, innerColBs=0,1  ← col2 无线程（需threadIdx.x=8，不存在）
  //   Bs 共 24 个元素（row0~7，col0~2）
  //
  //   floor strideB=2，写入地址 Bs[(innerRowGroupBs+rowIdx*2)*3+innerColBs]：
  //     rowIdx=0: thread0→Bs[0]=row0col0  thread1→Bs[1]=row0col1  thread2→Bs[2]=row0col2
  //               thread3→Bs[3]=row1col0  thread4→Bs[4]=row1col1  thread5→Bs[5]=row1col2
  //               thread6→Bs[6]=row2col0  thread7→Bs[7]=row2col1
  //     rowIdx=1: thread0→Bs[6]=row2col0(重写) thread1→Bs[7]=row2col1(重写) thread2→Bs[8]=row2col2
  //               thread3→Bs[9]=row3col0  thread4→Bs[10]=row3col1 thread5→Bs[11]=row3col2
  //               thread6→Bs[12]=row4col0 thread7→Bs[13]=row4col1
  //     rowIdx=2: thread0→Bs[12]=row4col0(重写) thread1→Bs[13]=row4col1(重写) thread2→Bs[14]=row4col2
  //               thread3→Bs[15]=row5col0 thread4→Bs[16]=row5col1 thread5→Bs[17]=row5col2
  //               thread6→Bs[18]=row6col0 thread7→Bs[19]=row6col1
  //   结果：row0~5 完整 ✓，row6 仅 col0,col1（Bs[20]=row6col2 未写）✗，row7 完全未写（Bs[21..23]）✗
  //
  //   ceil strideB=3，写入地址 Bs[(innerRowGroupBs+rowIdx*3)*3+innerColBs]：
  //     rowIdx=0: thread0→Bs[0] ... thread5→Bs[5]  thread6→Bs[6]=row2col0  thread7→Bs[7]=row2col1
  //     rowIdx=1: thread0→Bs[9]=row3col0 ... thread5→Bs[14]=row4col2
  //               thread6→Bs[15]=row5col0  thread7→Bs[16]=row5col1
  //     rowIdx=2: thread0→Bs[18]=row6col0 ... thread5→Bs[23]=row7col2 ✓
  //               thread6→Bs[24] 越界  thread7→Bs[25] 越界（Bs 共 24 个元素，索引 0~23）✗
  //   永远未写：Bs[8]=row2col2，Bs[17]=row5col2 → 结果错误 ✗
  //
  // ④ assert(BK * BN % blockDim.x == 0)
  //   作用：保证 Bs 的正确迭代次数是整数，与 ② 完全对称，对象换为 Bs
  //   前提：③ 已成立（strideB 是整数）
  //
  //   反例：BK=5, BN=4, blockDim.x=8
  //   strideB=8/4=2（整数 ✓），smemNum_exact=20/8=2.5
  //   Bs 共 20 个元素（row0~4），innerRowGroupBs：thread0-3→0，thread4-7→1
  //
  //   floor smemNum=2：
  //     rowIdx=0: thread0-3→Bs[0..3]=row0  thread4-7→Bs[4..7]=row1
  //     rowIdx=1: thread0-3→Bs[8..11]=row2 thread4-7→Bs[12..15]=row3
  //     row4（Bs[16..19]）永远未写 → 结果错误 ✗
  //
  //   ceil smemNum=3：
  //     rowIdx=0~1: 同上，row0~3 正确写入
  //     rowIdx=2: thread0-3→Bs[16..19]=row4 ✓
  //               thread4-7→Bs[20..23] ← 越界（Bs 共 20 个元素，索引 0~19）✗
  //   row4 正确加载，但 thread4-7 写 Bs[20..23]→ SMEM 溢出 ✗
  //
  // 注：代码用单个 smemNum（由 BM×BK 计算）同时驱动 As/Bs 两个循环，
  //     还需 assert(BM==BN) 保证该 smemNum 对 Bs 同样正确。
  //
  // ── "ceil smemNum + 行越界 guard" 能否替代四个 assert？ ──────────────────────
  //
  // 方案：ceil smemNum，写 As/Bs 前加 guard：
  //   if (innerRowGroupAs + rowIdx * strideA < BM) { As[...] = A[...]; }
  //   if (innerRowGroupBs + rowIdx * strideB < BK) { Bs[...] = B[...]; }
  //
  // 对 ②④ 有效（迭代次数不对）：
  //   以 BM=9, BK=4, blockDim.x=8, strideA=2, ceil smemNum=5 为例：
  //     rowIdx=4: thread0-3 写 row8（8<9 ✓）；thread4-7 的 row=9，9<9 为 false → 跳过
  //   → row8 正确加载，越界线程被 guard 拦截，结果完全正确 ✓
  //   assert②④ 可以去掉，改用 ceil+guard。
  //
  // 对 ①③ 无效（线程映射结构性残缺）：
  //   以 BK=3, blockDim.x=8, ceil strideA=3 为例，加 guard 后：
  //     rowIdx=0: thread6→As[6]=row2col0 ✓  thread7→As[7]=row2col1 ✓
  //               As[8]=row2col2 → 无任何线程写入（需 threadIdx.x=8，不存在）
  //     rowIdx=1: thread6→row5col0 ✓  thread7→row5col1 ✓
  //               As[17]=row5col2 → 同上，永久缺失
  //     rowIdx=2: thread6→row8，8<8 为 false → guard 跳过 ✓（防了越界）
  //   guard 只能拦截已有线程的越界写，无法凭空创造不存在的线程。
  //   As[8] 和 As[17] 始终是垃圾值 → 结果错误 ✗
  //   assert①③ 不可省略。
  //
  // 若要彻底去掉 assert①③，需改用平铺索引（flat indexing）：
  //   uint flat = threadIdx.x + rowIdx * blockDim.x;  // 线性编号，不依赖 BK 整除性
  //   if (flat < BM * BK) {
  //       uint row = flat / BK, col = flat % BK;
  //       As[row * BK + col] = A[(InitRow + row) * K + Idx + col];
  //   }
  //   每个 As 元素有唯一线程负责，BK 能否整除 blockDim.x 均正确。
  //   代价：每次迭代多一次除法和取模运算。
  //
  // ── guard 与 __syncthreads() 的兼容性 ────────────────────────────────────────
  //
  // guard（if 判断）不影响 __syncthreads()，两者完全兼容：
  //
  //   for (rowIdx ...) {
  //       if (row < BM) { As[...] = A[...]; }   ← 部分线程跳过写入
  //   }
  //   __syncthreads();   ← 所有线程仍能到达此处 ✓
  //
  // __syncthreads() 的唯一要求：它本身不能在发散分支内（即不能有线程到不了它）。
  // guard 只影响写入操作，__syncthreads() 在 if 外面，所有线程都能执行到，无死锁。
  //
  // 以下写法才有问题（__syncthreads() 在 if 内）：
  //   if (row < BM) {
  //       As[...] = A[...];
  //       __syncthreads();   ← ✗ 部分线程进不来 → 死锁
  //   }
  // ══════════════════════════════════════════════════════════════════════════════

  // 保证了每个block每一轮能处理完整的As的行
  assert(blockDim.x % BK == 0);
  // BM * BK % blockDim.x == 0保证了BM * BK == blockDim.x * smemNum == blockDim.x * BM * BK / blockDim.x,即保证了经过n次迭代刚好能全部处理完A的所有元素
  // assert(BM * BK == blockDim.x * smemNum);
  assert(BM * BK % blockDim.x == 0);
  // 保证了每个block每一轮能处理完整的Bs的行
  assert(blockDim.x % BN == 0);
  assert(BM == BN);
  //保证了经过n次迭代刚好能全部处理完B的所有元素
  assert(BK * BN % blockDim.x == 0);
  // 每个线程负责的SMEM的迁移个数，向上取整，即迭代的次数
  const uint smemNum {BM * BK / blockDim.x};
  // 每次迭代中，需要跨过的行数,因为一个block的线程能处理这么多行
  const uint strideA = blockDim.x / BK;
  const uint strideB = blockDim.x / BN;


  // 线程负责的SMEM元素的迁移的列和行组
  const uint innerRowGroupAs = threadIdx.x / BK;
  const uint innerColAs = threadIdx.x % BK;
  const uint innerRowGroupBs = threadIdx.x / BN;
  const uint innerColBs = threadIdx.x % BN;


  // 每个线程寄存器寄存乘积累加数组，处理TM * TN个数组
  float treadResultArr[TM * TN];
  // 循环K列
  for (uint Idx = 0; Idx < K; Idx+=BK) {
    // ── 填充节奏 ────────────────────────────────────────────────────────────────
    // 每轮（rowIdx）：blockDim.x 个线程同时各写 1 个 As 元素 + 1 个 Bs 元素
    // 循环 smemNum 轮后，整块 As（BM×BK 个）和 Bs（BK×BN 个）全部就绪
    //
    // ── As/Bs SMEM 写入地址连续性 ───────────────────────────────────────────────
    // As 写入地址化简：
    //   As[(innerRowGroupAs + rowIdx*strideA)*BK + innerColAs]
    //   = As[(threadIdx.x/BK)*BK + threadIdx.x%BK + rowIdx*strideA*BK]
    //   = As[threadIdx.x + rowIdx*blockDim.x]           ← 纯线性，每轮连续
    //
    // 各轮写入范围（以 BM=8, BK=4, blockDim.x=8 为例）：
    //   rowIdx=0 → As[0 .. 7 ]
    //   rowIdx=1 → As[8 .. 15]
    //   rowIdx=2 → As[16.. 23]
    //   rowIdx=3 → As[24.. 31]
    // Bs 同理，写入地址同样化简为 Bs[threadIdx.x + rowIdx*blockDim.x]，每轮连续。
    //
    // ── 全局内存读取连续性 ──────────────────────────────────────────────────────
    // 读 A：A[(InitRow + innerRowGroupAs + rowIdx*strideA)*K + Idx + innerColAs]
    //   warp 内不同 innerRowGroupAs 的线程访问不同行，行间距 = K×4 字节 → 非连续
    //   （即前述 4 次内存事务、带宽利用率 25% 的根本原因）
    //
    // 读 B：B[(innerRowGroupBs + rowIdx*strideB + Idx)*N + InitCol + innerColBs]
    //   同一 innerRowGroupBs 的线程 innerColBs 连续，访问同一行相邻列 → 连续 ✓
    // ────────────────────────────────────────────────────────────────────────────
    for (uint rowGroupIdx {}; rowGroupIdx < smemNum; ++rowGroupIdx) {
      // ── Global memory 读取的 coalesced 分析 ─────
      // cache line = 128 字节 = 32 float；sector = 32 字节 = 8 float
      //
      // A 全局读（非合并访问，带宽利用率 25%）：
      //   innerColAs = threadIdx.x % BK，warp 内每 BK=8 个线程共享同一行
      //   warp 32 线程分属 32/BK = 4 个不同行，行间距 = K × 4 字节
      //   每组 8 线程访问 32 字节（1 sector）
      //   K ≥ 32 时（典型 K=4096，行间距 16384 字节 >> cache line 128 字节）：
      //   → 4 组落在 4 条不同 cache line → 4 次内存事务 ✗
      //   带宽利用率：每条 cache line 加载 128 字节，每组只用 8 float = 32 字节
      //     合计加载：4 × 128 = 512 字节
      //     合计使用：4 ×  32 = 128 字节
      //     利用率：128 / 512 = 25% ✗
      //   误区："32线程×1float=128字节=1条cache line" 成立前提是32个float在同一cache line
      //         实际分散在4条cache line，GPU必须加载全部4条
      //
      // B 全局读（合并访问，带宽利用率 100%）：
      //   innerColBs = threadIdx.x % BN，BN ≥ 32，warp 内 threadIdx.x=0..31
      //   → innerColBs=0..31 连续，innerRowGroupBs 全为 0（同一行）
      //   → 32 线程的 32 个 float 恰好落在同一条 cache line（128 字节）
      //   → 加载 128 字节，使用 128 字节，带宽利用率 100% ✓ → 1 次内存事务
      //
      // ── SMEM 写入的 bank conflict 分析 ─────
      // As 写入：
      //   As[(innerRowGroupAs + rowIdx*strideA)*BK + innerColAs]
      //   warp 内 threadIdx.x=0..31 → 地址为 As[0], As[1], ..., As[31]
      //   → 32 个不同 bank，无 bank conflict ✓
      //
      // Bs 写入：
      //   Bs[(innerRowGroupBs + rowIdx*strideB)*BN + innerColBs]
      //   innerColBs = threadIdx.x（0..31），innerRowGroupBs = 0
      //   → 地址为 Bs[0], Bs[1], ..., Bs[31] → 32 个不同 bank，无 bank conflict ✓
      //
      // ── 三目表达式的作用 ─────────────────────────────────────────────────────
      // As[...] = (condition) ? A[...] : 0.0f
      //
      // 左侧 As[...] 的写入永远执行（与 SMEM 边界无关，四个 assert 保证 SMEM 索引合法）。
      // 三目只决定从全局内存读取真实值还是填 0.0f，防止读取矩阵 A/B 的越界地址。
      //
      // 触发 0.0f 的两种情况（均为全局矩阵边界，与 As/Bs 无关）：
      //   1. (InitRow + innerRowGroupAs + rowIdx*strideA) >= M
      //      → block 的行范围超出矩阵 A 的行数
      //      → M 不是 BM 的倍数时，最后一行的 block 会出现此情况
      //   2. Idx + innerColAs >= K
      //      → 当前 K 分块超出矩阵 A 的列数
      //      → K 不是 BK 的倍数时，最后一个 K tile 会出现此情况
      //   填 0.0f 作为 padding，不影响累加结果（加 0 不改变点积值）
      // 曾错误认为：三目表达式是为了防止 As/Bs SMEM 越界写入
      // 实际：As[...] 的 SMEM 写入始终发生，四个 assert 已保证 SMEM 索引合法；
      //       三目只防止从全局内存 A/B 越界读取（全局矩阵边界，与 SMEM 无关）
      As[(innerRowGroupAs + rowGroupIdx * strideA) * BK + innerColAs] = ((InitRow + innerRowGroupAs + rowGroupIdx * strideA) < M && Idx + innerColAs < K)? A[(InitRow + innerRowGroupAs + rowGroupIdx * strideA) * K + Idx + innerColAs]:0.0f;
      Bs[(innerRowGroupBs + rowGroupIdx * strideB) * BN + innerColBs] = ((innerRowGroupBs + rowGroupIdx * strideB + Idx) < K &&   (InitCol + innerColBs) < N)? B[(innerRowGroupBs + rowGroupIdx * strideB + Idx) * N  + InitCol + innerColBs]:0.0f;
    }
    // 等待所有线程写入SMEM完毕
    __syncthreads();



    // ── 计算阶段 SMEM bank conflict 分析（BM=BN=64, BK=8, TM=TN=8）──────────────
    // BN/TN = 64/8 = 8，warp 内线程分布：
    //   threadIdx.x  0.. 7 → threadRowGroup=0, threadColGroup=0..7
    //   threadIdx.x  8..15 → threadRowGroup=1, threadColGroup=0..7
    //   threadIdx.x 16..23 → threadRowGroup=2, threadColGroup=0..7
    //   threadIdx.x 24..31 → threadRowGroup=3, threadColGroup=0..7
    //
    // ── As 读取（4-way bank conflict）────────────────────────────────────────────
    // As[(threadRowGroup*TM + rowIdx)*BK + Idx]
    // 固定 rowIdx 和 Idx，4 个行组各 8 线程广播同一地址，4 个地址的 bank：
    //   r=0: As[rowIdx*8+Idx]         bank = (rowIdx*8+Idx) % 32
    //   r=1: As[64+rowIdx*8+Idx]      bank = (64+rowIdx*8+Idx) % 32 = 同上（64%32=0）
    //   r=2: As[128+rowIdx*8+Idx]     bank = 同上（128%32=0）
    //   r=3: As[192+rowIdx*8+Idx]     bank = 同上（192%32=0）
    // TM×BK = 8×8 = 64 = 2×32，步长是 32 的倍数 → 4 个地址落在同一 bank
    // → 4-way bank conflict ✗（组内广播无法消除跨组冲突）
    //
    // ── Bs 读取（2-way bank conflict）────────────────────────────────────────────
    // Bs[Idx*BN + threadColGroup*TN + colIdx]
    // 固定 Idx 和 colIdx，8 个列组各 4 线程广播，8 个唯一地址的 bank（BN=64，64%32=0）：
    //   j=0: (colIdx)%32     j=4: (32+colIdx)%32 = colIdx%32   ← 与 j=0 冲突
    //   j=1: (8+colIdx)%32   j=5: (40+colIdx)%32               ← 与 j=1 冲突
    //   j=2: (16+colIdx)%32  j=6: (48+colIdx)%32               ← 与 j=2 冲突
    //   j=3: (24+colIdx)%32  j=7: (56+colIdx)%32               ← 与 j=3 冲突
    // {j=0,4}{j=1,5}{j=2,6}{j=3,7} 各 2 个地址同一 bank → 2-way bank conflict ✗
    //
    // 注：参数不同结果不同（BN=128 时 As=2-way, Bs=4-way；BN=64 时 As=4-way, Bs=2-way）
    //     两种情况 As 和 Bs 均存在 bank conflict，根本原因是 TM×BK 和 TN 与 32 的整除关系
    //     kernel_7（resolve_bank_conflicts）通过 SMEM padding 或转置解决此问题
    // ────────────────────────────────────────────────────────────────────────────
    for (uint Idx {}; Idx < BK; ++Idx) {
      float rowTemp[TM];
      for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
        rowTemp[rowIdx] = As[(threadRowGroup * TM +rowIdx) * BK  + Idx];
      }

      float colTemp[TN];
      for (uint colIdx {}; colIdx < TN; ++colIdx) {
        colTemp[colIdx] = Bs[Idx * BN + threadColGroup * TN + colIdx];
      }

      for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
        for (uint colIdx {}; colIdx < TN; ++colIdx) {
          // treadResultArr 是每线程私有的 TM×TN 二维结果，按行主序存为一维数组：
          //   逻辑布局（TM=3, TN=4）：
          //            col0 col1 col2 col3
          //     row0 [  0    1    2    3  ]
          //     row1 [  4    5    6    7  ]
          //     row2 [  8    9   10   11  ]
          //   物理存储：index = rowIdx * TN + colIdx
          //     rowIdx * TN：跳过前 rowIdx 整行，每行宽度 = TN（线程负责的列数）
          //     colIdx    ：列内偏移
          //   stride 用 TN 而非 BK/BN，因为该数组只属于当前线程，宽度就是 TN
          // 曾错误写成：treadResultArr[(threadRowGroup + rowIdx) * BK + threadColGroup + colIdx]
          //   错误①：stride 用 BK（SMEM 宽度），实际应为 TN（本地数组宽度）
          //   错误②：混入 threadRowGroup/threadColGroup（这两个是线程在 block tile 中的位置，
          //           用于写入 C，不用于本地私有数组——本地数组每线程独立，行组从 0 开始）
          treadResultArr[rowIdx * TN + colIdx] += rowTemp[rowIdx] * colTemp[colIdx];
        }
      }
    }
    // 等待所有线程写入SMEM完毕
    __syncthreads();
  }

  // ── 写入 C 的 coalesced 分析（BM=BN=64, TM=TN=8）──────────────────────────────
  // warp 内线程分布（BN/TN=8）：
  //   threadIdx.x  0.. 7 → threadRowGroup=0, threadColGroup=0..7
  //   threadIdx.x  8..15 → threadRowGroup=1, threadColGroup=0..7
  //   threadIdx.x 16..23 → threadRowGroup=2, threadColGroup=0..7
  //   threadIdx.x 24..31 → threadRowGroup=3, threadColGroup=0..7
  //
  // 固定 (rowIdx, colIdx)，4 个行组写入 4 条不同行：
  //   行组 r 写入行 InitRow + r*TM + rowIdx
  //   相邻行组行间距 = TM × N × 4 字节（TM=8, N=4096 时 = 131072 字节 >> cache line 128 字节）
  //   → 4 行组地址落在 4 段完全不同的内存区域，行间无法合并
  //
  // 同一行组内 8 个线程的列位置（stride = TN = 8）：
  //   InitCol+colIdx, +8, +16, +24, +32, +40, +48, +56
  //   → cache line 0（32 floats）：命中 +0, +8, +16, +24  → 4/32 有效
  //   → cache line 1（32 floats）：命中 +32,+40,+48,+56  → 4/32 有效
  //   每行组 2 次事务，4 行组共 8 次事务
  //
  //   有效写入：32 floats = 128 字节
  //   实际访问：8 × 128 = 1024 字节
  //   带宽利用率：128 / 1024 = 12.5% ✗
  //
  // 根本原因：相邻线程列距 = TN=8（不连续），跨行组时地址跳跃 TM×N×4 字节
  // 对比 kernel_4：warp 内 32 线程同行且列连续 → 1 次事务，100% ✓
  // ────────────────────────────────────────────────────────────────────────────

  // 写入C
  // 曾错误写成：C[(InitRow + threadRowGroup + rowIdx) * BN + InitCol + threadColGroup + colIdx]
  //   错误①：threadRowGroup 缺少 * TM（行组起始行 = 行组编号 × 每组行数）
  //   错误②：threadColGroup 缺少 * TN（列组起始列 = 列组编号 × 每组列数）
  //   错误③：stride 用 BN（block tile 宽度），实际应为 N（全局矩阵列宽）
  for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
    for (uint colIdx {}; colIdx < TN; ++colIdx) {
      if ((InitRow + threadRowGroup * TM + rowIdx) < M &&  InitCol + threadColGroup * TN + colIdx < N) {
        C[(InitRow + threadRowGroup * TM + rowIdx) * N + InitCol + threadColGroup * TN + colIdx] = alpha * treadResultArr[rowIdx * TN + colIdx] + beta * C[(InitRow + threadRowGroup * TM + rowIdx) * N + InitCol + threadColGroup * TN + colIdx];
      }
    }
  }


}
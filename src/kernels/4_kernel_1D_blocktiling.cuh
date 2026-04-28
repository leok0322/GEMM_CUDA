#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ══════════════════════════════════════════════════════════════════════════════
// kernel_4：1D Blocktiling
//
// 核心改进（对比 kernel_3）：
//   每线程从计算 1 个 C 元素 → 计算 TM 个（M 方向连续），引入 tmpB 寄存器缓存
//
// ── 关键优化：tmpB 寄存器缓存 Bs ──────────────────────────────────────────
//
//   kernel_3（每线程 1 个 C 元素）：
//     for dotIdx in BK:
//       tmp += As[row][dotIdx] * Bs[dotIdx][col]   // Bs 每次从 smem 读，无复用
//
//   kernel_4（每线程 TM 个 C 元素，M 方向）：
//     for dotIdx in BK:
//       tmpB = Bs[dotIdx][threadCol]               // 读 1 次，缓存到寄存器
//       for resIdx in TM:                          //   threadCol 固定 → 地址不变
//         threadResults[resIdx] += As[threadRow*TM+resIdx][dotIdx] * tmpB
//         // As 随 resIdx 变化 → 必须读 TM 次（无法缓存）
//         // tmpB 地址不变 → 寄存器复用 TM 次，省 TM-1 次 smem 读
//   每 dotIdx 节省 TM-1 次 Bs smem 读；BK 次迭代共节省 (TM-1)×BK 次/tile
//
//   对称版本（N 方向 tiling）：每线程固定 1 行 × TM 列，改为缓存 tmpA：
//     for dotIdx in BK:
//       tmpA = As[threadRow][dotIdx]               // 固定，缓存到寄存器
//       for colIdx in TM:
//         threadResults[colIdx] += tmpA * Bs[dotIdx][threadCol*TM+colIdx]
//   kernel_5（2D blocktiling）同时缓存 tmpA 和 tmpB，双向复用
//
// ── 模板参数与线程数 ────────────────────────────────────────────────────────
//
//   BM = 64, BN = 64, BK = 8, TM = 8
//
//   blockDim.x = BM*BN/TM = 64*64/8 = 512，同时满足：
//     BM*BK = 64*8 = 512（加载 As 所需线程数）
//     BN*BK = 64*8 = 512（加载 Bs 所需线程数）
//   三者相等 → 每个线程恰好负责加载 As/Bs 各 1 个元素（assert 验证）
//
// ── Shared Memory ──────────────────────────────────────────────────────────
//
//   As: BM×BK = 64×8 = 512 floats = 2 KiB（矩形，非方形，因 BK≠BM）
//   Bs: BK×BN = 8×64 = 512 floats = 2 KiB
//   合计 4 KiB（kernel_3 为 8 KiB，差异来自 BK=8 而非 BLOCKSIZE=32）
//
// ── 寄存器 ────────────────────────────────────────────────────────────────
//
//   CUDA kernel 中所有局部变量（标量、小数组）默认分配到寄存器；
//   cRow/cCol/threadCol/threadRow 等坐标也在寄存器，不在 global memory。
//   local memory（global memory 的一块）只在寄存器溢出时才使用。
//
//   48 个寄存器的大致拆解（ptxas 实测，0 bytes spill）：
//     threadResults[0..7]  : 8   ← TM=8 个累加器（主要目的，显式占用）
//     tmpB                 : 1   ← 内层循环 Bs 缓存
//     cRow, cCol           : 2   ← block tile 行列坐标
//     threadCol, threadRow : 2   ← 计算阶段线程坐标
//     innerRowA/ColA       : 2   ← 加载 As 的线程坐标
//     innerRowB/ColB       : 2   ← 加载 Bs 的线程坐标
//     bkIdx, dotIdx, resIdx: 3   ← 三层循环计数器
//     A, B, C 指针         : 6   ← 64-bit 地址各占 2 个 32-bit 寄存器
//     地址偏移中间量        : ~22 ← 乘加地址计算的临时结果
//     ────────────────────────────
//     合计约 48 个
//
//   每 block 寄存器：512×48 = 24576；SM 总量 65536
//   → 每 SM 最多 2 个 block（寄存器是 occupancy 瓶颈，smem 仅 4 KiB/block 不是）
//
//   TM 较大时 threadResults 可能溢出（spill）到 local memory（~300 cycles/access）；
//   TM=8 时 8 个 float 完全在寄存器内，无溢出。
//
// ── Grid / Block ────────────────────────────────────────────────────────────
//
//   gridDim = (CEIL_DIV(N,BN), CEIL_DIV(M,BM))  ← x→列，y→行（与 kernel_3 相反）
//   blockDim = (512, 1, 1)；每个 block 负责 C 的一个 BM×BN = 64×64 tile
// ══════════════════════════════════════════════════════════════════════════════
template <const int BM, const int BN, const int BK, const int TM>
__global__ void gemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {

  // ── blockIdx 方向：x→列（cCol），y→行（cRow）──────────────────────────────
  // 与 kernel_3 相反（kernel_3：cRow=blockIdx.x，cCol=blockIdx.y），性能提升约 30%
  //
  // 原因：GPU 调度时 blockIdx.x 变化最快（linear_ID = blockIdx.y*gridDim.x + blockIdx.x）
  //   → 连续 block 的 blockIdx.x 递增、blockIdx.y 不变
  //   → 相邻 block 访问相同 A 行区域，A tile 连续存储，L2 可高效复用
  //
  // 数值对比（M=N=K=4096, BM=BN=64，grid=64×64）：
  //
  //   ✓ kernel_4（cRow=blockIdx.y, cCol=blockIdx.x）——A 行复用：
  //   blockID  blockIdx.x  blockIdx.y  cRow  cCol  A 行范围    B 列范围
  //      0         0           0        0     0    0.. 63     0.. 63
  //      1         1           0        0     1    0.. 63    64..127  ← A 行相同，block 0 将 1MB A tile 加载进 L2
  //      2         2           0        0     2    0.. 63   128..191  ← block 1~63 直接命中，A 高效复用 ✓
  //
  //   ✗ 若反转（cRow=blockIdx.x, cCol=blockIdx.y）——B 列"复用"（效率低）：
  //   blockID  blockIdx.x  blockIdx.y  cRow  cCol  A 行范围    B 列范围
  //      0         0           0        0     0    0.. 63     0..63
  //      1         1           0        1     0   64..127     0..63   ← B 列相同，但 B 行间距 N=4096 float=16KB
  //      2         2           0        2     0  128..191     0..63   ← B tile 散落 64MB 地址范围，L2 命中率低 ✗
  //                                            A 各 block 行范围不同，无复用，每次读 1MB ✗
  //
  // 根本原因：A tile（连续行）内存紧凑，L2 友好；B tile（行间距大）地址分散，L2 不友好
  //   → 复用 A 远比复用 B 有价值
  //
  // 【Global memory 物理访问单位】
  //   L2 命中：以 sector（32 字节 = 8 float）为单位返回数据
  //   L2 未命中：以 cache line（128 字节 = 4 sector）为单位从 DRAM 加载，再返回所需 sector
  //   coalesced 访问（warp 读连续 32 float = 128 字节）：
  //     1 次 DRAM 事务取 128 字节，全部有用，带宽利用率 100% ✓
  //   跨步访问（32 线程各访问不同 cache line）：
  //     最多 32 次 DRAM 事务，每次取 128 字节，实际只用 4 字节，利用率 3% ✗
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // ── 计算阶段的线程坐标（C tile 内的位置）────────────────────────────────
  // 【关键】threadCol → 直接是列下标；threadRow → 行组编号（需 ×TM 才是实际行号）
  //
  // blockDim.x = BM*BN/TM 个线程，映射到 (BM/TM)×BN 的二维网格（行主序）：
  //   列维度：BN = 64；行维度：BM/TM = 8 个行组
  //
  // 1D→2D 行主序：col = id % 列数，row = id / 列数（除数是列数 BN，不是行数 BM/TM）
  //   直觉：连续 id 先填满 BN 个列才换行 → 每隔 BN 个 id 行号才 +1
  //
  // threadCol = threadIdx.x % BN → 0..63（C tile 列下标，直接用于访问 Bs 和写回 C）
  //   同一 threadCol 出现 BM/TM 次（每列对应 BM/TM 个线程，负责该列的不同行组）
  //
  // threadRow = threadIdx.x / BN → 0..7（行组编号，不是实际行号！）
  //   范围推导：blockDim.x = BM/TM*BN，最大 threadIdx.x = BM/TM*BN-1
  //     threadRow_max = (BM/TM*BN-1)/BN = BM/TM-1 = 7 ✓
  //   实际行号 = threadRow * TM + resIdx（resIdx 在计算循环中 0→TM-1）
  //   该线程处理行范围：[threadRow*TM, threadRow*TM+TM-1]
  //
  // 不对称性：threadCol 直接用（每线程 1 列，无需换算）；threadRow 需 ×TM（每线程 TM 行）
  //   设计原因：每线程固定 1 列（负责 TM 行），列宽 = BN → 用 BN 做模/除
  //   若改为固定 1 行（负责 TM 列）：threadRow = threadIdx.x % BM，threadCol = threadIdx.x / BM
  //   错误写法：threadIdx.x / (BM/TM) = threadIdx.x / 8 → 结果 0..63，超行组上限 7 ✗
  //             （除数必须是列数 BN，不是行数 BM/TM）
  //
  // 与加载坐标的关系：innerColB = threadIdx.x % BN，innerRowB = threadIdx.x / BN
  //   公式与 threadCol/threadRow 完全相同，因为 Bs 列宽 = C tile 列宽 = BN
  //
  // 示例（BN=64, TM=8）：
  //   threadIdx.x=0:  threadCol=0, threadRow=0 → 行 0*8+resIdx=0..7,  列0
  //   threadIdx.x=1:  threadCol=1, threadRow=0 → 行 0*8+resIdx=0..7,  列1
  //   threadIdx.x=64: threadCol=0, threadRow=1 → 行 1*8+resIdx=8..15, 列0
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // ── Shared Memory ──────────────────────────────────────────────────────
  // As: BM×BK = 64×8（矩形，非方形，因 BK≠BM）
  // Bs: BK×BN = 8×64（矩形，非方形，因 BK≠BN）
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // ── 将指针推进到本 block 负责区域的起始位置 ──────────────────────────
  // A += cRow*BM*K（跳过 cRow*BM 行，每行 K 元素）
  // B += cCol*BN  （跳过 cCol*BN 列，在第 0 行内偏移）
  // C += cRow*BM*N + cCol*BN
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // assert：验证 BM*BK = BN*BK = blockDim.x，确保每线程恰好负责加载 As/Bs 各 1 个元素
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);

  // ── 加载阶段的线程坐标（协作加载 As/Bs，与计算阶段坐标独立）──────────────
  //
  // 1D→2D 行主序：col = id % 列数，row = id / 列数（除数是列数，不是行数）
  //
  // As（BM×BK，列数=BK=8）：
  //   innerColA = threadIdx.x % BK → 0..7  = 0..BK-1 ✓
  //   innerRowA = threadIdx.x / BK → 0..63 = 0..BM-1 ✓
  //   注意：不是 /BM（除以 64 结果只有 0..7，覆盖不了 BM=64 行）✗
  //   coalesced：BK=8，warp 内每 8 线程访问同一行连续 8 元素（64 B），需 4 次事务
  //
  // Bs（BK×BN，列数=BN=64）：
  //   innerColB = threadIdx.x % BN → 0..63 = 0..BN-1 ✓
  //   innerRowB = threadIdx.x / BN → 0..7  = 0..BK-1 ✓
  //   注意：不是 /BK（除以 8 结果是 0..63，超过 Bs 的 BK=8 行范围）✗
  //   coalesced：BN=64，warp 内 threadIdx.x=0..31 → innerColB=0..31 连续 → 完全 coalesced ✓
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  // TM 个 C 元素的部分和累加器，全部驻寄存器（512 线程×8 = 4096 个/block）
  // threadResults[resIdx] 对应 C tile 内 (threadRow*TM + resIdx, threadCol) 位置
  float threadResults[TM] = {0.0};

  // ── 主循环：沿 K 方向逐 tile 累加 ─────────────────────────────────────
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

    // 协作加载：每线程加载 As/Bs 各一个元素
    // As[innerRowA][innerColA] ← A[innerRowA][bkIdx + innerColA]
    // Bs[innerRowB][innerColB] ← B[bkIdx + innerRowB][innerColB]
    // （A/B 指针已推进到本 block 起始，直接用相对偏移；此版本不做边界检查）
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // 推进指针到下一个 K tile
    A += BK;      // A 沿列方向推进 BK 列
    B += BK * N;  // B 沿行方向推进 BK 行

    // ── 点积计算：dotIdx 外层，resIdx 内层 ─────────────────────────────
    // 【dotIdx 必须在外层】：
    //   固定 dotIdx 时 Bs[dotIdx*BN+threadCol] 地址不变 → 读一次存 tmpB，复用 TM 次
    //   若 resIdx 在外层：每个 resIdx 迭代重新读 Bs → TM 倍 smem 读，收益消失
    //
    // Bs bank conflict（结论：无）：
    //   warp 内 threadIdx.x=0..31 → threadCol=0..31（BN=64，连续不回绕）
    //   Bs[dotIdx*BN+0..31]：同一行连续 32 float → 32 个不同 bank ✓
    //
    // As bank conflict（结论：无）：
    //   resIdx 是串行循环变量（SIMD 步进），不构成并发访问，不存在 bank conflict
    //   同一时刻 warp 内 threadIdx.x=0..31，threadRow = threadIdx.x/BN，BN=64 > 32
    //   → 全部 /64 = 0 → threadRow 均为 0 → 32 线程访问同一地址 → SMEM 广播 ✓
    //   （若 BN < 32，如 BN=8：不同线程 threadRow 不同 → 有 bank conflict）
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Bs 固定地址读一次，缓存到寄存器；As 地址随 resIdx 变化，每次从 smem 读
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();  // 防止快线程进入下一 tile 提前覆盖 As/Bs
  }

  // ── 写回结果：每线程写 TM 个 C 元素 ────────────────────────────────────
  // C tile 内行 = threadRow*TM + resIdx，列 = threadCol
  //
  // coalesced：warp 内 threadCol=0..31 连续，(threadRow*TM+resIdx) 相同
  //   → C 同一行连续 32 个元素，1 次 DRAM 事务 ✓
  //
  // 示例（threadIdx.x=0，threadRow=0, threadCol=0，TM=8）：
  // ┌────────┬────────────────────────┬────────┐
  // │ resIdx │ 行 = threadRow*TM+resIdx│ 列     │
  // ├────────┼────────────────────────┼────────┤
  // │   0    │         0              │   0    │
  // │   1    │         1              │   0    │
  // │  ...   │        ...             │  ...   │
  // │   7    │         7              │   0    │
  // └────────┴────────────────────────┴────────┘
  // → 该线程写了行 0~7、列 0 共 8 个元素——1D Blocktiling 的核心：每线程负责 TM 行
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// gemm1DBlocktiling_v2：自行实现的 1D Blocktiling kernel
//
// 【相比 kernel_3（gemm_shared_mem_block）的改进】
//
// 1. 每线程计算 TM 个 C 元素（M 方向）
//    kernel_3：每线程 1 个 C 元素，BLOCKSIZE² 个线程/block
//    v2：每线程 TM=8 个 C 元素，BM*BN/TM = 512 个线程/block
//
// 2. tmpCol 寄存器缓存，减少 shared memory 读取
//    kernel_3：每次点积都从 smem 读 Bs，无复用
//    v2：Bs[colIdx*BN+threadCol] 读入寄存器 tmpCol，在 TM 次 rowIdx 迭代中复用
//    收益：每个 colIdx 省 TM-1 次 smem 读，共节省 (TM-1)*BK 次/tile
//
// 3. threadResults[TM] 寄存器累加器
//    kernel_3：单个 temp 寄存器；v2：TM 个累加器全驻寄存器，无 smem 中间写回
//
// 4. 完整的 M/N/K 三方向边界检查
//    kernel_3 v1：无 M/N 边界检查，非整数倍维度时越界读
//    v2：加载用三元掩码（越界填 0），写回逐行 if 过滤，__syncthreads() 在 if 外
//
// ══════════════════════════════════════════════════════════════════════════════
// 【编写过程中的典型错误总结】
//
// 错误类1：block 编号与元素索引混淆
//   initRowGroup/initColGroup 是 block 编号，不是元素位置
//   全局行 = initRowGroup * BM + innerROWForAs（必须乘以 BM）
//   全局列 = initColGroup * BN + innerCOLForBs（必须乘以 BN）
//
// 错误类2：SMEM 步长用错矩阵维度
//   As 是 BM×BK → 步长 = BK：As[innerROWForAs * BK + innerCOLForAs]
//   Bs 是 BK×BN → 步长 = BN：Bs[innerROWForBs * BN + innerCOLForBs]
//   曾错误地用 BK 作为 Bs 的步长
//
// 错误类3：B 的行列方向混淆
//   B 是 K×N 矩阵：行方向 = K，列方向 = N，步长 = N
//   曾在 B 行号中混入 M 方向的 initRowGroup（B 与 M 无关）
//   曾将步长写成 K（应为 N）
//
// 错误类4：__syncthreads() 在条件 if 内部 → 死锁
//   同一 block 内不同线程边界条件不同，if 内的 syncthreads 只被部分线程调用
//   正确做法：加载和两个 __syncthreads() 必须在任何 if 外，所有线程都执行
//
// 错误类5：循环结构错误
//   = + 写成赋值而非 +=，导致只保留最后一次 colIdx 的结果
//   写回放在 K tile 循环内，导致 beta 被多次乘（应在循环外统一写回）
//   第二个 __syncthreads() 遗漏，导致快线程覆盖慢线程还在读的 As/Bs
//
// 错误类6：边界条件不完整
//   threadRowGroup 是行组编号，边界检查需乘以 TM 才得起始行
//   innerROWForAs 和 threadRowGroup 是同一 threadIdx.x 的两套独立映射
//   写回检查不能保护 A 加载，两者各自独立保护不同内存访问
// ══════════════════════════════════════════════════════════════════════════════
template <const int BM, const int BN, const int BK, const int TM>
__global__ void gemm1DBlocktiling_v2(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // 这个block起始的行和列
  const uint initRowGroup {blockIdx.y};
  const uint initColGroup {blockIdx.x};

  // 当前线程负责的行组和列
  const uint threadCol {threadIdx.x % BN};
  // 因为一个block的线程数为BM * BN / TM，所以有BM/TM个行组，也就是说每个线程要处理TM行
  const uint threadRowGroup {threadIdx.x / BN};

  // 静态SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];


  // 分配每个线程处理的SMEM数组中的元素
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerROWForAs {threadIdx.x / BK};
  const uint innerCOLForAs {threadIdx.x % BK};
  const uint innerROWForBs {threadIdx.x / BN};
  const uint innerCOLForBs {threadIdx.x % BN};


  // 一个BLOCK处理BM*BN个元素
  float threadResults[TM] {};
  for (int idx {};idx <  CEIL_DIV(K, BK); ++idx) {
    // ── As 加载 ──────────────────────────────────────────────────────────
    // 【曾犯错误】A 全局行号写成 (initRow + innerROWForAs)
    //   initRow = blockIdx.y 是 block 编号，不是元素行号
    //   block编号 × block行数 才是起始元素行：initRow * BM + innerROWForAs
    // 【边界检查：A 加载需独立的 M 方向检查】
    // innerROWForAs = threadIdx.x/BK 与 threadRowGroup = threadIdx.x/BN 是两套独立映射：
    //   BK < BN，所以 innerROWForAs 增长更快，可能超过 threadRowGroup*TM
    //   例：threadIdx.x=72, BK=8, BN=64, TM=8
    //     threadRowGroup*TM = (72/64)*8 = 8
    //     innerROWForAs     = 72/8      = 9  → 若 M=9，则 9 不 < 9，A 越界！
    // 写回检查（threadRowGroup*TM+rowIdx）保护 C，但不保护 A 加载，两者相互独立
    As[innerROWForAs * BK + innerCOLForAs] = ((innerCOLForAs + (idx * BK))< K  && (initRowGroup * BM + innerROWForAs) < M)? A[(initRowGroup * BM + innerROWForAs) * K + innerCOLForAs + (idx * BK)]:0.0f;

    // ── Bs 加载 ──────────────────────────────────────────────────────────
    // 【曾犯错误1】SMEM 写成 Bs[innerROWForBs * BK + ...]
    //   Bs 是 BK×BN 矩阵，行步长 = 列数 = BN，不是 BK
    //
    // 【曾犯错误2】B 全局行号含 initRow：(initRow + innerROWForBs + idx*BK)
    //   B 是 K×N 矩阵，行方向是 K，与 M 方向的 initRow 无关
    //   正确：innerROWForBs + idx * BK
    //
    // 【曾犯错误3】B 全局步长写成 * K
    //   B 行主序，每行 N 个元素，步长 = N
    //
    // 【曾犯错误4】B 全局列号写成 initCol + innerCOLForBs
    //   initCol = blockIdx.x 是 block 编号，需 × BN 换算为元素列号
    //   正确：initCol * BN + innerCOLForBs
    Bs[innerROWForBs * BN + innerCOLForBs] = ((innerROWForBs + (idx * BK))< K && (initColGroup * BN +innerCOLForBs) < N)? B[(innerROWForBs + (idx * BK)) * N + initColGroup * BN +innerCOLForBs]:0.0f;
    // 同步
    __syncthreads();


    // 一个BK长度小矩阵的乘积累加
    for (int colIdx {}; colIdx < BK; ++colIdx) {
      // 固定列，处理该列的As的行组与固定的列元素乘积
      // 每个线程处理TM个行，所属的行组是threadRowGroup，列是colIdx
      // 取Bs中的固定threadCol，所在的行就是colIdx
      // WARP中32个线程访问BS,因为Bs的列BN为64，不会bankconflict
      float tmpCol {Bs[colIdx * BN +threadCol]};
      // 固定列的情况下，循环行
      for (int rowIdx {}; rowIdx < TM; ++rowIdx) {
        //一个WARP读取AS的同一个元素，广播
        // 每个线程计算TM个行元素
        threadResults[rowIdx] += As[(threadRowGroup * TM + rowIdx)* BK + colIdx] * tmpCol;
      }
    }
    // 第二次同步：防止快线程进入下一轮 idx 循环提前覆盖 As/Bs，
    // 而慢线程还在读当前 tile 的 As/Bs（threadResults 是私有寄存器，无需同步）
    __syncthreads();
  }
  for (int rowIdx {}; rowIdx < TM; ++rowIdx) {
    // 【写回边界检查：逐行检查，不能省略】
    // threadRowGroup*TM 是起始行，rowIdx 从 0 跑到 TM-1
    // 起始行合法不代表全部 TM 行合法（M 不能整除 TM 时最后一组会有部分越界）
    //   例：M=9, threadRowGroup=1, TM=8 → 起始行8 < 9 ✓，但 rowIdx=1 时行9 >= 9 越界
    // 此检查保护 C 的写入；A 加载的 M 方向检查是独立的，两者各自保护不同内存访问

    // 【曾犯错误】外层边界条件写成：
    //   (initRowGroup * BM + threadRowGroup) < M
    //   threadRowGroup 是行组编号（0..BM/TM-1），不是行号，需乘以 TM 才是起始行
    //   正确：(initRowGroup * BM + threadRowGroup * TM + rowIdx) < M
    if (initRowGroup * BM + threadRowGroup * TM + rowIdx < M && initColGroup * BN + threadCol < N) {
      // 写入C，每个线程写入TM个元素
      // WARP写入C数组连续，合并访问事务
      C[(initRowGroup * BM + threadRowGroup * TM + rowIdx) * N + initColGroup * BN + threadCol] = alpha * threadResults[rowIdx] + beta *  C[(initRowGroup * BM + threadRowGroup * TM + rowIdx) * N + initColGroup * BN + threadCol];
    }
  }
}

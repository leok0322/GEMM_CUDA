#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 【kernel_4：1D Blocktiling】
//
// 核心改进（对比 kernel_3）：
//   kernel_3：每个线程计算 C 的 1 个元素，1024 线程/block
//   kernel_4：每个线程计算 C 的 TM 个元素（M 方向连续），512 线程/block

// 关键优化：tmpB 寄存器缓存，减少 shared memory 读取
//
//   kernel_3 的内层循环（每个线程算 1 个 C 元素）：
//     for dotIdx in BK:
//       tmp += As[row][dotIdx] * Bs[dotIdx][col]   // As/Bs 各读 1 次，无复用
//
//   kernel_4 的内层循环（每个线程算 TM 个 C 元素，M 方向）：
//     for dotIdx in BK:
//       tmpB = Bs[dotIdx][threadCol]               // 读 1 次，缓存到寄存器
//       for resIdx in TM:
//         threadResults[resIdx] += As[(threadRow*TM+resIdx)][dotIdx] * tmpB
//         // As：resIdx 每次不同行 → 必须读 TM 次（无法缓存）
//         // tmpB：dotIdx/threadCol 固定 → 寄存器复用 TM 次，省 TM-1 次 smem 读
//
//   每个 dotIdx 迭代：Bs 读 1 次（缓存）vs TM 次（无缓存）→ 节省 (TM-1)×BK 次 Bs 读
//
// 对称性（也可以缓存 As）：
//   若每个线程改为负责 1 行 × TM 列（N 方向 tiling），则可缓存 tmpA：
//     for dotIdx in BK:
//       tmpA = As[threadRow][dotIdx]               // 读 1 次，缓存到寄存器
//       for colIdx in TM:
//         threadResults[colIdx] += tmpA * Bs[dotIdx][threadCol*TM+colIdx]
//         // tmpA 复用 TM 次，省 TM-1 次 As 读
//   kernel_4 选择 M 方向（缓存 Bs），对称版本是 N 方向（缓存 As），效果相同
//   kernel_5（2D blocktiling）同时做两个方向，缓存 tmpA 和 tmpB，复用翻倍

// 模板参数（实际值来自 runner.cu）：
//   BM = 64：block 在 M 方向覆盖的行数
//   BN = 64：block 在 N 方向覆盖的列数
//   BK =  8：block 在 K 方向每次处理的列/行数（tile 深度）
//   TM =  8：每个线程在 M 方向计算的元素数（thread-level tiling）
//
// 线程数：blockDim.x = BM * BN / TM = 64 * 64 / 8 = 512
//   同时满足：BM * BK = 64 * 8 = 512（加载 As 所需线程数）
//             BN * BK = 64 * 8 = 512（加载 Bs 所需线程数）
//   assert 保证三者相等，确保每个线程恰好负责加载 As/Bs 各一个元素
//
// shared memory：
//   As: BM × BK = 64 × 8 = 512 floats = 2 KiB
//   Bs: BK × BN = 8 × 64 = 512 floats = 2 KiB
//   合计 4 KiB（kernel_3 为 8 KiB，更小是因为 BK=8 而非 32）
//
// 每个线程的寄存器：
//   threadResults[TM=8]：存储 TM 个 C 元素的累加器（寄存器数组）
//   512 线程 × 8 寄存器 = 4096 个累加器寄存器/block
//
// grid/block 划分：
//   gridDim = (CEIL_DIV(N,BN), CEIL_DIV(M,BM))  ← 注意 x/y 与 kernel_3 相反
//   blockDim = (BM*BN/TM, 1, 1) = (512, 1, 1)
//   每个 block 负责 C 的一个 BM×BN = 64×64 的 tile

template <const int BM, const int BN, const int BK, const int TM>
__global__ void gemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {

  // ── blockIdx 方向与 kernel_3 相反：x→列，y→行 ──────
  // kernel_3：cRow=blockIdx.x，cCol=blockIdx.y
  // kernel_4：cRow=blockIdx.y，cCol=blockIdx.x  ← 刻意交换
  //
  // 原因：交换后性能提升约 30%
  //   GPU 调度 block 时 blockIdx.x 变化最快：
  //     linear_ID = blockIdx.y * gridDim.x + blockIdx.x
  //     → 连续 block ID 对应 blockIdx.x = 0,1,2,...，blockIdx.y 不变
  //
  // 【数值举例：M=N=K=4096, BM=BN=64，grid=64×64】
  //
  //   ✓ kernel_4（cRow=blockIdx.y, cCol=blockIdx.x）：
  //
  //   blockID | blockIdx.x | blockIdx.y | cRow | cCol | A 行范围   | B 列范围
  //      0    |     0      |     0      |  0   |  0   |  0.. 63  |   0.. 63
  //      1    |     1      |     0      |  0   |  1   |  0.. 63  |  64..127  ← A 行相同，B 列不同
  //      2    |     2      |     0      |  0   |  2   |  0.. 63  | 128..191  ← A 行相同，B 列不同
  //
  //   A tile（行0..63 全部K列）：64行×4096列×4B = 1MB，连续存储
  //   → block 0 将这 1MB 加载进 L2，block 1~63 直接命中 → A 高效复用
  //   B tile：每个 block 列范围不同，无 cross-block 复用（但无妨，见下文）
  //
  //   ✗ 若反过来（cRow=blockIdx.x, cCol=blockIdx.y）：
  //
  //   blockID | blockIdx.x | blockIdx.y | cRow | cCol | A 行范围   | B 列范围
  //      0    |     0      |     0      |  0   |  0   |  0.. 63  |  0..63
  //      1    |     1      |     0      |  1   |  0   | 64..127  |  0..63   ← B 列相同，A 行不同
  //      2    |     2      |     0      |  2   |  0   |128..191  |  0..63   ← B 列相同，A 行不同
  //
  //   B tile（全部K行，列0..63）：看似可复用，但 B 是行主序存储：
  //     每行的列0..63 与下一行的列0..63 相距 N=4096 个 float = 16KB
  //     4096行 × 16KB 间距 → B tile 数据散落在 64MB 地址范围内
  //     → L2 cache line 极度分散，"复用"效率极低
  //   A tile：每个 block 行范围不同，无复用，且每次都要读新的 1MB
  //   → 两者均无高效复用 → 慢 30%
  //
  // 【根本原因：A tile 连续，B tile 分散】
  //   A[cRow*BM..(cRow+1)*BM-1][:]：BM 个连续行 → 内存中紧凑的连续块 → L2 友好
  //   B[:][cCol*BN..(cCol+1)*BN-1]：每行取 BN 列，行间距 N float = 16KB → 地址分散 → L2 不友好
  //   因此复用 A（kernel_4）远比复用 B（反转）更有价值

  //
  // 【Global memory 访问的物理单位：sector 与 cache line】
  //
  //   L2 命中（数据已在缓存中）：
  //     以 sector（32字节 = 8个float）为单位返回数据
  //     warp 请求哪些 sector，L2 直接返回，不访问 DRAM
  //
  //   L2 未命中（数据不在缓存中）：
  //     以 cache line（128字节 = 4个sector）为单位从 DRAM 加载到 L2
  //     再将所需 sector 返回给线程
  //     其余 sector 也留在 L2，后续访问若能复用则值得，否则是带宽浪费
  //
  //   coalesced 访问的意义（以 Bs 加载为例，warp 读连续 32 个 float = 128字节）：
  //     L2 未命中：1次 DRAM 事务取 128字节（1 cache line = 4 sector），全部有用
  //     带宽利用率 128/128 = 100% ✓
  //
  //   若跨步访问（32线程各访问不同 cache line）：
  //     L2 未命中：最多 32次 DRAM 事务，每次取 128字节，实际只用 4字节
  //     带宽利用率 4/128 = 3%，浪费 97% ✗
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // ── 计算阶段的线程坐标（对应 C tile 内的位置）───────
  // blockDim.x = BM*BN/TM = 512，线程一维排列
  // 每个线程负责 C tile 内固定1列（threadCol）× TM 行（由 threadRow 行组决定）
  //
  // 线程被映射到一个 (BM/TM) × BN 的二维网格（行主序展开）：
  //   行维度：BM/TM = 8 个行组（每组覆盖 TM=8 行）
  //   列维度：BN = 64 个列
  //
  // 1D→2D 行主序公式：col = id % 列数，row = id / 列数 → 除数是【列数】，不是行数
  //   直觉：行主序中连续 id 先填满一行的所有列（BN 个），填完才换行
  //         → 每隔 BN 个 id，行号才 +1 → 必须除以列数 BN
  //
  // threadCol = threadIdx.x % BN：该线程负责的【列号】，0..BN-1 = 0..63
  //   threadCol 直接就是 C tile 内的列下标，无需换算，直接用于写回和访问 Bs
  //   同一 threadCol 值出现 BM/TM 次（每列对应 BM/TM 个线程，各负责该列的不同行组）
  //
  // threadRow = threadIdx.x / BN：该线程所属的【行组编号】，0..BM/TM-1 = 0..7
  //   范围推导：blockDim.x = BM/TM * BN，故 threadIdx.x 最大为 BM/TM*BN - 1
  //     threadRow_max = (BM/TM*BN - 1) / BN = BM/TM - 1 = 7 ✓
  //   threadRow 是行组编号，不是实际行号，需要换算：
  //   实际行号 = threadRow * TM + resIdx（resIdx 在计算循环中从 0 遍历到 TM-1）
  //   该线程处理的实际行范围：[threadRow*TM, threadRow*TM+TM-1]
  //
  // threadCol 与 threadRow 不对称：
  //   threadCol → 直接是列下标（每线程 1 列，无需 ×TN）
  //   threadRow → 行组编号（每线程 TM 行，需要 ×TM + resIdx 才得实际行号）
  //
  // 注意：不是 threadIdx.x / (BM/TM)
  //   threadIdx.x / 8 → 结果范围 0..63，远超行组上限 BM/TM-1=7 ✗
  //   原因：(BM/TM) 是行数，行主序中除数必须是列数 BN，不是行数
  //
  // 为什么用 BN 不用 BM？
  //   因为设计是"每线程固定1列（沿 N 方向），负责 TM 行"：
  //     列有 BN=64 个可能值 → 列维度宽度 = BN → 用 BN 做模/除
  //   若改为"每线程固定1行，负责 TN 列"（N 方向 tiling），则反过来：
  //     threadRow = threadIdx.x % BM（行数做模），threadCol = threadIdx.x / BM
  //
  // 与 innerColB/innerRowB 的关系：
  //   innerColB = threadIdx.x % BN，innerRowB = threadIdx.x / BN
  //   → 与 threadCol/threadRow 公式完全相同，因为 B tile 列宽 = C tile 列宽 = BN
  //
  // 示例（BN=64, TM=8）：
  //   threadIdx.x=0:   threadCol=0,  threadRow=0（行组0）→ 实际行 0*8+resIdx = 0..7,  列0
  //   threadIdx.x=1:   threadCol=1,  threadRow=0（行组0）→ 实际行 0*8+resIdx = 0..7,  列1
  //   threadIdx.x=64:  threadCol=0,  threadRow=1（行组1）→ 实际行 1*8+resIdx = 8..15, 列0
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // ── shared memory 分配 ────────────
  // As: BM×BK = 64×8，存储 A 当前 tile（矩形，非方形）
  // Bs: BK×BN = 8×64，存储 B 当前 tile（矩形，非方形）
  // kernel_3 的 As/Bs 是方形（BLOCKSIZE×BLOCKSIZE），此处因 BK≠BM/BN 变为矩形
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // ── 将指针推进到本 block 负责区域的起始位置 ───────
  // A：跳过 cRow*BM 行（每行 K 个元素）→ A += cRow*BM*K
  // B：跳过 cCol*BN 列（在第 0 行内偏移）→ B += cCol*BN
  // C：跳过 cRow*BM 行 + cCol*BN 列 → C += cRow*BM*N + cCol*BN
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // ── assert：验证线程数与 tile 大小匹配 ────────────
  // 每个线程加载 As 的 1 个元素，需要 BM*BK 个线程
  // 每个线程加载 Bs 的 1 个元素，需要 BK*BN 个线程
  // blockDim.x 必须同时满足两个要求，因此 BM*BK = BN*BK = blockDim.x
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);

  // ── 加载阶段的线程坐标（用于协作加载 As/Bs，与计算阶段坐标不同）───────────
  //
  // As 加载坐标（BM×BK 视角）：
  //   As 是 BM×BK = 64×8 的矩阵，需要把线性 threadIdx.x（0..511）映射为 (row, col)
  //
  //   一维→二维的通用公式（行主序）：
  //     col = id % 列数     row = id / 列数   ← 除数是列数，不是行数
  //
  //   As 列数 = BK = 8：
  //     innerColA = threadIdx.x % BK = threadIdx.x % 8  → 0..7  = 0..BK-1 ✓
  //     innerRowA = threadIdx.x / BK = threadIdx.x / 8  → 0..63 = 0..BM-1 ✓
  //
  //   为什么不是 threadIdx.x / BM（除以行数）？
  //     threadIdx.x / 64 → 结果只有 0..7，覆盖不了 64 行 ✗
  //     必须除以列数 BK=8，步长才与列宽匹配，行号才能跑满 0..63
  //
  //   → 512 线程覆盖 As 的全部 BM*BK=512 个元素，一一对应
  //
  //   coalesced 分析（BK=8）：
  //   warp 内 32 个线程，innerColA = threadIdx.x % 8，innerRowA = threadIdx.x / 8
  //   → 线程 0..7 访问 A[row0][0..7]，线程 8..15 访问 A[row1][0..7]，...
  //   → 每组 8 线程访问同一行连续 8 个元素（64 bytes），但 4 组行不同
  //   → 需要 4 次内存事务（非完全 coalesced），但硬件可合并相邻事务
  //
  // Bs 加载坐标（BK×BN 视角）：
  //   Bs 是 BK×BN = 8×64 的矩阵，列数 = BN = 64：
  //     innerColB = threadIdx.x % BN = threadIdx.x % 64 → 0..63 = 0..BN-1 ✓
  //     innerRowB = threadIdx.x / BN = threadIdx.x / 64 → 0..7  = 0..BK-1 ✓
  //
  //   为什么不是 threadIdx.x / BK（除以行数）？
  //     threadIdx.x / 8 → 结果是 0..63，远超 Bs 的 8 行范围 ✗
  //     必须除以列数 BN=64，行号才能恰好覆盖 0..7
  //
  //   → 512 线程覆盖 Bs 的全部 BK*BN=512 个元素，一一对应
  //
  //   coalesced 分析（BN=64）：
  //   warp 内 32 个线程，innerColB = threadIdx.x % 64 = 0..31（连续）
  //   innerRowB = threadIdx.x / 64 = 0（全 warp 相同）
  //   → 访问 B 同一行连续 32 个元素 → 完全 coalesced ✓
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  // ── 每个线程的寄存器累加器：存储 TM 个 C 元素的部分和 ────────────────────
  // threadResults[resIdx] 对应 C tile 内 (threadRow*TM + resIdx, threadCol) 位置
  // 共 TM=8 个，全部存在寄存器中（不进 shared memory，不进 global memory）
  // 512 线程 × 8 = 4096 个累加器，均摊到寄存器堆



  // 【48 个寄存器都存了什么？——所有局部变量默认都在寄存器中】
  //
  // 误区：cRow/cCol/threadCol/threadRow 不在 global memory 中，而是在寄存器里
  //   CUDA kernel 中所有局部变量（标量、小数组）默认分配到寄存器
  //   global memory 只存放：传入的矩阵指针所指向的 A/B/C 数组本身
  //   "local memory"（global memory 的一块）只在寄存器溢出（spill）时才使用
  //   本 kernel：0 bytes spill stores, 0 bytes spill loads → 无溢出，所有局部量在寄存器
  //
  // 48 个寄存器的组成（近似拆解）：
  //   threadResults[0..7]  : 8 个  ← TM=8 个累加器（主要目的，显式占用）
  //   tmpB                 : 1 个  ← 内层循环 B tile 缓存
  //   cRow, cCol           : 2 个  ← block tile 行列坐标
  //   threadCol, threadRow : 2 个  ← 计算阶段线程坐标
  //   innerRowA/ColA       : 2 个  ← 加载 As 的线程坐标
  //   innerRowB/ColB       : 2 个  ← 加载 Bs 的线程坐标
  //   bkIdx, dotIdx, resIdx: 3 个  ← 三层循环计数器
  //   A, B, C 指针         : 各 2 个（64-bit 地址占 2 个 32-bit 寄存器）= 6 个
  //   地址偏移中间量        : ~10 个 ← 各种乘加地址计算的临时结果
  //   ─────────────────────────────
  //   合计约 48 个（ptxas 实测值）
  //
  // 什么时候需要寄存器：
  //   所有 kernel 内的局部变量（标量/小数组）→ 编译器自动分配寄存器
  //   循环计数器、指针、中间计算值 → 均在寄存器
  //
  // 什么时候不用寄存器（或寄存器不够用时）：
  //   1. __shared__ 变量：显式放在 shared memory，不占寄存器
  //   2. 寄存器溢出（spill）：局部变量太多超过每线程上限（255个）时
  //      编译器将部分变量存到 local memory（实为 global memory）
  //      ptxas 报告 "N bytes spill stores/loads"，每次访问代价 ~300 cycles
  //   3. 大数组：threadResults[TM] 若 TM 很大，编译器可能将其放入 local memory
  //      TM=8 时 8 个 float 完全可以放入寄存器 → 无溢出
  //
  // 寄存器数量对 occupancy 的影响（sm_86）：
  //   每个 SM 共 65536 个寄存器，每个线程 48 个
  //   每 block 512 线程 × 48 = 24576 个寄存器/block
  //   65536 / 24576 ≈ 2.67 → 每个 SM 最多同时驻留 2 个 block
  //   （shared memory 4 KiB/block，100 KiB / 4 KiB = 25，不是瓶颈）
  //   → 寄存器是此 kernel 的 occupancy 瓶颈
  float threadResults[TM] = {0.0};

  // ── 主循环：沿 K 方向逐 tile 累加 
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

    // ── 协作加载：每线程加载 As/Bs 各一个元素 ─────
    // As[innerRowA][innerColA] ← A[innerRowA][bkIdx + innerColA]
    // Bs[innerRowB][innerColB] ← B[bkIdx + innerRowB][cCol*BN + innerColB]
    // （A/B 指针已推进，直接用相对偏移）
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // ── 推进指针到下一个 k_tile ───
    A += BK;        // A 沿列方向推进 BK 列
    B += BK * N;    // B 沿行方向推进 BK 行

    // ── 点积计算：dotIdx 为外层，resIdx 为内层 ────
    // 【关键优化：dotIdx 外层，tmpB 缓存在寄存器，被 TM 个线程结果复用】
    //
    // 对每个 dotIdx（BK 方向）：
    //   tmpB = Bs[dotIdx][threadCol]：从 shared memory 读一次，存入寄存器
    //   对 TM 个 resIdx：
    //     threadResults[resIdx] += As[threadRow*TM+resIdx][dotIdx] * tmpB
    //     ↑ As 每次从 shared memory 读，共读 TM 次（不同行）
    //     ↑ tmpB 直接从寄存器读，复用 TM 次（0 次 shared memory 访问）
    //
    // 若 dotIdx 为内层（resIdx 外层）：
    //   每个 resIdx 迭代都需要重新从 shared memory 读 tmpB → TM 倍读取开销
    //   现在 dotIdx 外层：tmpB 只读一次，复用 TM 次 → shared memory 读减少 TM 倍
    //
    // Bs bank conflict 分析：
    //   warp 内所有线程 threadCol = threadIdx.x % BN，BN=64 > 32
    //   warp 内 threadIdx.x=0..31 → threadCol=0..31（连续）
    //   Bs[dotIdx * BN + threadCol]：dotIdx 相同，threadCol=0..31 连续
    //   → 访问同一行连续 32 个 float → 32 个不同 bank → 无 bank conflict ✓
    //
    // As bank conflict 分析（结论：无 bank conflict）：
    //
    // bank conflict 的前提是：同一 warp 内不同线程在【同一时刻】访问同一 bank 的不同地址。
    // resIdx 是 for 循环变量，每次迭代是串行（SIMD 步进），不是并发访问，不构成 bank conflict。
    //
    // 分析 warp 内各线程在同一时刻的 As 访问地址：
    //   地址 = As[(threadRow * TM + resIdx) * BK + dotIdx]
    //   threadRow = threadIdx.x / BN，BN=64 > warp_size=32
    //   → warp 内 threadIdx.x=0..31 全部 / 64 = 0 → threadRow 均为 0
    //   → 同一 resIdx、dotIdx 时，warp 内 32 个线程访问同一地址
    //   → SMEM 广播（broadcast），无 bank conflict ✓
    //
    // 若 BN < warp_size（如 BN=8）：
    //   threadIdx.x=0..31 → threadRow=0,0,..,0,1,1,..,3（每 8 个线程 threadRow 递增 1）
    //   不同线程 threadRow 不同 → 地址间距 TM*BK=64 bytes → 不同 bank 组合 → 有 bank conflict
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 【为什么 Bs 缓存到寄存器，而 As 不用？】
      //
      // 固定 dotIdx，看内层 resIdx 循环的访问地址：
      //   Bs[dotIdx * BN + threadCol]：dotIdx 固定，threadCol 固定
      //     → TM 次迭代访问的是同一个地址 → 常量 → 读一次存 tmpB，复用 TM=8 次
      //   As[(threadRow * TM + resIdx) * BK + dotIdx]：resIdx 每次 +1
      //     → TM 次迭代访问 TM 个不同地址 → 变量 → 无论如何都要读 TM 次，缓存无意义
      //
      // 缓存 tmpB 的收益（per dotIdx 迭代）：
      //   Bs shared memory 读：1次（缓存）vs 8次（不缓存）→ 省 7 次
      //   As shared memory 读：8次（不变）
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // ── 写回结果：每个线程写 TM 个 C 元素 ──────
  // (threadRow*TM + resIdx) 是 C tile 内的行，threadCol 是 C tile 内的列
  // resIdx=0..TM-1：连续 TM 行，每次写一个元素
  //
  // coalesced 分析：
  //   warp 内 threadCol=0..31 连续，(threadRow*TM+resIdx) 相同
  //   → C 同一行连续 32 个元素 → coalesced ✓

  // 对 threadIdx.x=0（threadRow=0, threadCol=0，TM=8）：
  // ┌────────┬────────────────────────────────────┬──────────┐
  // │ resIdx │ 写入的行 = threadRow × TM + resIdx │ 写入的列    │
  // ├────────┼────────────────────────────────────┼──────────┤
  // │ 0      │ 0×8+0 = 行 0                       │ 列 0      │
  // ├────────┼────────────────────────────────────┼──────────┤
  // │ 1      │ 0×8+1 = 行 1                       │ 列 0      │
  // ├────────┼────────────────────────────────────┼──────────┤
  // │ 2      │ 0×8+2 = 行 2                       │ 列 0      │
  // ├────────┼────────────────────────────────────┼──────────┤
  // │ ...    │ ...                                │ ...      │
  // ├────────┼────────────────────────────────────┼──────────┤
  // │ 7      │ 0×8+7 = 行 7                       │ 列 0      │
  // └────────┴────────────────────────────────────┴──────────┘
  // 所以 threadIdx.x=0 确实写了行 0~7、列 0，共 8 个元素，这正是 1D Blocktiling 的核心：每个线程不再只算 1 个 C 元素，而是算 TM=8 个（沿 M 方向连续的 8 行）。threadResults[TM] 这 8 个累加器对应的就是这 8 行。
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}


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

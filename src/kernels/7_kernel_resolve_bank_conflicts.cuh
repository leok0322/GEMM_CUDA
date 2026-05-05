#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ══════════════════════════════════════════════════════════════════════════════
// kernel_7 改进：对 Bs 进行重排（linearize），消除计算阶段读取 Bs 的 bank conflict
//
// ── 与 kernel_6 的唯一区别 ───────────────────────────────────────────────────
//
//   kernel_6：Bs 行主序存储，计算阶段读 Bs 有 4-way bank conflict
//   kernel_7：Bs 写入时重排，计算阶段读 Bs 无 bank conflict
//
// ── Bs 重排的映射规则 ────────────────────────────────────────────────────────
//
//   B[r][c] → Bs[new_row * 16 + new_col]
//     new_row = r * 8 + c % 8   （原始行 × BK + 列内偏移）
//     new_col = c / 8            （列组号，∈ {0..15}）
//
//   写入公式（innerRowB=r，innerColB=c/4，k=c%4）：
//     Bs[((innerColB%2)*4 + innerRowB*8 + k) * 16 + innerColB/2] = tmp.{k}
//
//   读取公式（dotIdx=r，i=colIdx，threadCol=c/8）：
//     regN[i] = Bs[(dotIdx*8 + i) * 16 + threadCol]
//
//   验证：new_row = dotIdx*8+i = r*8+c%8 ✓，new_col = threadCol = c/8 ✓
//
// ── 读取 bank conflict 消除分析 ──────────────────────────────────────────────
//
//   kernel_6 行主序读：regN[i] = Bs[dotIdx*BN + threadCol*TN + i]
//     threadCol 步长 = TN=8，周期 = 32/8 = 4 → 每 4 个 threadCol bank 重复
//     → 4-way conflict
//
//   kernel_7 重排后读：regN[i] = Bs[(dotIdx*8+i)*16 + threadCol]
//     固定 dotIdx/i，threadCol=0..15 对应地址步长=1 → 16 个不同 bank
//     threads 16..31 的 threadCol 与 0..15 相同 → broadcast（不是 conflict）
//     → 无 conflict ✓
//
// ── 为何 kernel_7 实测反而比 kernel_6 慢（4377 vs 4699 GFLOPS）────────────
//
//   直觉：消除了 Bs 读 conflict → 应该更快
//   实测：反而更慢。原因在于对两个阶段代价的错误估计。
//
//   ① 【修正】kernel_6 Bs 写入实际是 0 conflict（不是 4-way）
//
//     kernel_6 写入：reinterpret_cast<float4*>(&Bs[...])[0] = tmp
//       4 个 float 写到连续地址 → 编译器生成 STS.128（128-bit 向量 store 指令）
//       STS.128 由硬件分 4 个 phase 执行，每 phase 8 个线程 × 4 float：
//         Phase 0：thread  0-7  → float  0-31  → bank 0-31（各 1 次）→ 0 conflict
//         Phase 1：thread  8-15 → float 32-63  → bank 0-31（各 1 次）→ 0 conflict
//         Phase 2：thread 16-23 → float 64-95  → bank 0-31（各 1 次）→ 0 conflict
//         Phase 3：thread 24-31 → float 96-127 → bank 0-31（各 1 次）→ 0 conflict
//       每 phase 8 线程 × 4 float = 32 个元素，恰好覆盖全部 32 bank 各一次
//       → kernel_6 Bs 写入 = 0 bank conflict
//
//     这是关键前提：kernel_7 的"优化"起点并非 4-way 写 conflict，而是 0 conflict
//
//   ② kernel_7 Bs 写入：从 0 冲突 → 引入新冲突 + 指令数 ×4
//
//     kernel_7 写入：4 条独立 scalar store，目标地址步长 = BN/TN = 16（非连续）
//       编译器无法合并为向量指令 → 4 条 STS.32，且：
//       - 2-way bank conflict（步长 16，每 2 个线程落同一 bank）
//       - 指令数从 1 增加到 4，加载阶段耗时延长
//
//     净写入变化：0 conflict，1 条指令 → 2-way conflict，4 条指令（双重劣化）
//
//   ③ 读 conflict 消除的实际收益接近 0（在 compute-bound 下）
//
//     kernel_6 计算阶段读 Bs 有 4-way conflict，但：
//       计算阶段有 BK=8 次外层循环，每次内层有 TM×TN=64 次 FMA
//       某 warp 读 conflict → stall → 调度器切换到其他 warp 执行 FMA
//       FMA 单元持续工作，conflict 延迟被完全掩盖
//       → compute-bound 时，读 conflict 的实际代价 ≈ 0
//
//     kernel_7 消除读 conflict，但消除的是一个代价已近乎为零的问题
//     → 读 conflict 消除的收益 ≈ 0
//
//   ④ 加载阶段 vs 计算阶段的代价不对称
//
//     计算阶段 conflict（kernel_6 读 conflict）：
//       被 __syncthreads__ 后面大量 FMA 工作掩盖 → 代价 ≈ 0
//
//     加载阶段 conflict / 指令增加（kernel_7 写 conflict + 4×指令）：
//       加载阶段结束后立刻是 __syncthreads__，全 block 等齐后才开始 FMA
//       加载阶段内无 FMA 可填充 stall → conflict 代价完全暴露
//
//       kernel_6：[──加载（0 conflict）──]──sync──[────FMA（4-way 读 conflict≈0）────]
//       kernel_7：[────加载（2-way + ×4指令）────]──sync──[────FMA（0 读 conflict）────]
//                      ↑ 这段额外时间 FMA 完全等待，无法隐藏
//
//   ⑤ 结论：
//     kernel_6 Bs 写入：STS.128 → phase → 0 conflict，1 条指令（加载阶段代价低）
//     kernel_7 Bs 写入：4×STS.32 → 2-way conflict，4 条指令（加载阶段代价高）
//     kernel_6 Bs 读取：4-way conflict，但 FMA 掩盖，实际代价 ≈ 0
//     kernel_7 Bs 读取：0 conflict，但收益 ≈ 0（消除了一个几乎免费的代价）
//     净效果：写入劣化（硬代价） > 读取优化（≈0 收益） → kernel_7 反而慢
//
//   ┌──────────────┬──────────────────────────────┬──────────────────────┬──────────────┐
//   │              │ Bs 写入（加载阶段）          │ Bs 读（计算阶段）    │ 实测         │
//   ├──────────────┼──────────────────────────────┼──────────────────────┼──────────────┤
//   │ kernel_6     │ STS.128，phase→0 conflict    │ 4-way（FMA掩盖≈0）  │ 4699 GFLOPS  │
//   │              │ 1 条指令                     │                      │              │
//   ├──────────────┼──────────────────────────────┼──────────────────────┼──────────────┤
//   │ kernel_7     │ 4条 STS.32，2-way conflict   │ 无                   │ 4377 GFLOPS  │
//   │              │ 4 条指令（4×）               │ （收益≈0）           │              │
//   └──────────────┴──────────────────────────────┴──────────────────────┴──────────────┘
//
//   正确消除 Bs bank conflict 的方法是 kernel_8 的 padding 方案：
//   保留 float4 向量写入（保留 STS.128），仅追加 padding 列消除读 conflict
// ══════════════════════════════════════════════════════════════════════════════

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN),1)  gemmResolveBankConflicts(int M, int N, int K, float alpha,
                                          float *A, float *B, float beta,
                                          float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

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
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // "linearize" Bs while storing it
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
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
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN),1) gemmResolveBankConflicts_v2(int M, int N, int K, float alpha,
                                          float *A, float *B, float beta,
                                          float *C) {
  // 该block的起始行和列
  const uint initRow {blockIdx.y * BM};
  const uint initCol {blockIdx.x * BN};

  // 该block的SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // BLOCK中该线程负责的行组和列组
  static_assert(BN % TN == 0 && "需要处理的列组数不是整数");
  // 列方向
  assert(blockDim.x % (BN / TN) == 0 && "block的线程不能完整覆盖行组中的完整列组");
  // 行方向
  assert(BM * BN % (blockDim.x * TN * TM) == 0 && "block的线程不能完整覆盖完整行组");
  const uint threadRowGroup {threadIdx.x / (BN / TN) };
  const uint threadColGroup {threadIdx.x % (BN / TN) };
  // 存储该线程需要的TM行元素进行复用
  float tempAs[TM];
  // 存储该线程需要的TN列元素进行复用
  float tempBs[TN];
  // 存储计算结果
  float threadResult [TM * TN] {0.0f};

  // 该线程负责的加载As和Bs的行和列组
  static_assert(BK % 4 == 0 && "As的向量加载的列不是整数");
  assert(blockDim.x % (BK / 4) == 0 && "block的线程不能完整覆盖As行的完整列组");
  assert(BM * BK % (blockDim.x * 4) == 0 && "block的线程不能完整覆盖As的完整行");
  static_assert(BN % 4 == 0 && "Bs的向量加载的列不是整数");
  assert(blockDim.x % (BN / 4) == 0 && "block的线程不能完整覆盖Bs行的完整列组");
  static_assert(BM == BN);
  const uint innerRowAs { threadIdx.x / (BK / 4) };
  const uint innerColGroupAs { threadIdx.x % (BK / 4) };
  const uint innerRowBs {threadIdx.x / (BN / 4) };
  const uint innerColGroupBs { threadIdx.x % (BN / 4) };
  const uint rowNumPerForAs { blockDim.x / (BK / 4) };
  const uint rowNumPerForBs { blockDim.x / (BN / 4) };
  // 该线程负责写入As的向量
  float4 vecAs;
  // 该线程负责写入的Bs的向量
  float4 vecBs;

  // Bs（BK，BN）加载重排
  static_assert(BN % TN == 0 && "Bs中的一列不包含整数个TN");
  assert((BK * BN) % (blockDim.x * TN)  == 0 && "一个block的线程能覆盖完整的Bs的行");


  // 向量写回C
  assert(TN % 4 == 0);
  float4 tmp;


  for (uint outterColIdx {}; outterColIdx < K; outterColIdx+=BK) {
    for (uint rowAs{}; rowAs < BM; rowAs += rowNumPerForAs) {
      vecAs = reinterpret_cast<float4 *>(&A[(initRow + innerRowAs + rowAs) * K + outterColIdx + innerColGroupAs * 4])[0];
      // 行主序
      // As[(innerRowAs + rowAs)  * BK + innerColGroupAs * 4]  = vecAs.x;
      // As[(innerRowAs + rowAs) * BK + innerColGroupAs * 4 + 1]  = vecAs.y;
      // As[(innerRowAs + rowAs) * BK + innerColGroupAs * 4 + 2]  = vecAs.z;
      // As[(innerRowAs + rowAs) * BK + innerColGroupAs * 4 + 3]  = vecAs.w;

      // 列主序
      As[(innerColGroupAs * 4) * BM + innerRowAs + rowAs] = vecAs.x;
      As[(innerColGroupAs * 4  + 1) * BM + innerRowAs + rowAs] = vecAs.y;
      As[(innerColGroupAs * 4  + 2) * BM + innerRowAs + rowAs] = vecAs.z;
      As[(innerColGroupAs * 4  + 3) * BM + innerRowAs + rowAs] = vecAs.w;
    }


    for (uint rowBs{}; rowBs < BK; rowBs += rowNumPerForBs) {
      // 行主序
      // BN=128, innerRowBs=threadIdx.x / (BN / 4)=0,0,...,0,0，innerColGroupBs=threadIdx.x % (BN / 4)=0，1，2，3，4，...，31
      // 一个warp32个线程，访问B的同一行
      vecBs = reinterpret_cast<float4 *>(&B[(innerRowBs + outterColIdx + rowBs) * N + innerColGroupBs * 4 + initCol])[0];
      // 读取B的时候：
      //   一个线程取4个float32，就是16字节，2个线程就是32个字节，并且同一行。线程对使用一个sector。
      //   所有的32个线程在同一行，元素连续，所以可以合并访问事务，8个线程共用一个cache line，所以32个线程一共4个访问事务
      // 写入Bs的时候：
      //   线程0写（0，0），线程1写（0，4），...
      //   那么，间隔4个元素，，每 8 个线程 bank 重复，所以4-way bank conflict
      // Bs 写入 bank conflict 分析：
      //   线程 t（warp 0，innerRowB=0）写入 Bs[t*4 .. t*4+3]（步长=4 个 float）
      //   若按标量分析：thread 0→bank 0，thread 8→bank 0（32=8×4）→ 4-way conflict
      //
      //   但此处写入是 float4 向量 store → 编译器生成 STS.128（128-bit SMEM store）
      //   STS.128 由硬件分 4 个 phase 执行，每 phase 8 个线程：
      //     Phase 0：thread  0-7  → float  0-31 → bank 0-31（各 1 次）→ 无冲突
      //     Phase 1：thread  8-15 → float 32-63 → bank 0-31（各 1 次）→ 无冲突
      //     Phase 2：thread 16-23 → float 64-95 → bank 0-31（各 1 次）→ 无冲突
      //     Phase 3：thread 24-31 → float 96-127→ bank 0-31（各 1 次）→ 无冲突
      //   每 phase 8 线程 × 4 float = 32 个元素，恰好覆盖全部 32 bank 各一次 → 实际 0 conflict
      // Bs[(innerRowBs + rowBs) * BN + innerColGroupBs * 4]  = vecBs.x;
      // Bs[(innerRowBs + rowBs) * BN + innerColGroupBs * 4 + 1]  = vecBs.y;
      // Bs[(innerRowBs + rowBs) * BN + innerColGroupBs * 4 + 2]  = vecBs.z;
      // Bs[(innerRowBs + rowBs) * BN + innerColGroupBs * 4 + 3]  = vecBs.w;

      // 一个warp32个线程，innerColGroupBs / 2= 0，0，1，1，2，2，...，15，15  innerColGroupBs % 2 * 4= 0，4，0，4，0，4，...，0，4， innerRowBs* TN = 0,0,...,0,0
      // 2-Way bank conflict
      // 相比kernel6不重排，冲突更少了，但是 performance: (4377.1) GFLOPS小于，kernel6 的performance: (4698.7) GFLOPS
      Bs[((innerColGroupBs % (TN/4)) * 4 + innerRowBs * TN + rowBs * TN) * (BN / TN)  + (innerColGroupBs / 2)]  = vecBs.x;
      Bs[((innerColGroupBs % (TN/4)) * 4 + innerRowBs * TN + rowBs * TN + 1) * (BN / TN)  + (innerColGroupBs / 2)]  = vecBs.y;
      Bs[((innerColGroupBs % (TN/4)) * 4 + innerRowBs * TN + rowBs * TN + 2) * (BN / TN)  + (innerColGroupBs / 2)] = vecBs.z;
      Bs[((innerColGroupBs % (TN/4)) * 4 + innerRowBs * TN + rowBs * TN + 3) * (BN / TN)  + (innerColGroupBs / 2)]  = vecBs.w;

    }

    // 同步
    __syncthreads();






    // 计算该线程负责的TM * TN个元素的结果
    for (uint innerColIdx{}; innerColIdx < BK; ++innerColIdx) {
      for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
        // 行主序
        // tempAs[rowIdx] = As[(threadRowGroup * TM + rowIdx)* BK + innerColIdx];

        // 列主序
        tempAs[rowIdx] = As[innerColIdx * BM + threadRowGroup * TM + rowIdx];
      }

      for (uint colIdx {}; colIdx < TN; ++colIdx) {
        // 读取Bs的时候的bank conflict分析：
        //   Bs（BK, BN），BN=128,TN=8, threadColGroup=threadIdx.x % (BN / TN)=threadIdx.x % 16
        //   一个warp的32个线程，0，1*8，2*8，...，15*8，0，1*8，2*8，...，15*8，线程间相差8个元素，每两个线程访问共一个元素，16 / 4 = 4-way conflict
        // tempBs[colIdx] = Bs[innerColIdx * BN + threadColGroup * TN + colIdx];

        // Bs重排后
        // 一个warp的32个线程，0，1，2，...，15，0，1，2，...，15，每2个线程访问共一个元素，无bank conflict
        tempBs[colIdx] = Bs[(innerColIdx * TN  + colIdx) * (BN / TN) + threadColGroup];
      }

      // 存储的计算结果复用
      for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
        for (uint colIdx {}; colIdx < TN; ++colIdx) {
          threadResult[rowIdx * TN + colIdx] += tempAs[rowIdx] * tempBs[colIdx];
        }
      }
    }

    __syncthreads();
  }

  // 将该线程的计算结果写回C,标量写入
  // for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
  //   for (uint colIdx {}; colIdx < TN; ++colIdx) {
  //     // 标量写入
  //     C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroup * TN + colIdx] = threadResult[rowIdx * TN + colIdx];
  //   }
  // }

  // 将该线程的计算结果写回C,向量写入
  for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
    for (uint colIdx {}; colIdx < TN; colIdx+=4) {
      tmp =  reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroup * TN + colIdx])[0];
      tmp.x = alpha * threadResult[rowIdx * TN + colIdx] + beta *  tmp.x;
      tmp.y = alpha * threadResult[rowIdx * TN + colIdx + 1]+ beta *  tmp.y;
      tmp.z = alpha * threadResult[rowIdx * TN + colIdx + 2]+ beta *  tmp.z;
      tmp.w = alpha * threadResult[rowIdx * TN + colIdx + 3]+ beta *  tmp.w;
      reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroup * TN + colIdx])[0] = tmp ;
    }
  }

}
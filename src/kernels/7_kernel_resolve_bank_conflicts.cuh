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
//   表面看消除了 Bs 读 conflict 应该更快，但两个因素导致性能下降：
//
//   ① Bs 写入：1 条向量指令 → 4 条标量指令（throughput 问题，无法隐藏）
//
//     kernel_6 写入：reinterpret_cast<float4*>(...)[0] = tmp
//       4 个 float 写到连续地址 → 编译器生成 1 条 st.shared.v4.f32
//
//     kernel_7 写入：4 条独立 scalar store，目标地址步长=16（非连续）
//       编译器无法生成向量指令 → 4 条 st.shared.f32，指令数增加 4 倍
//
//   ② 指令数增加的性能损失机制：加载阶段变长，FMA 等待 syncthreads（不是争抢发射槽）
//
//     常见误解：store 指令多了，占用了本该给 FMA 的发射槽
//     实际原因：加载阶段和计算阶段被 __syncthreads__ 严格隔开，不重叠：
//
//       加载阶段：GMEM→SMEM（store 指令在此执行）
//       __syncthreads()    ← 全 block 等齐后才能继续
//       计算阶段：SMEM→寄存器→FMA（FMA 指令在此执行）
//
//     store 指令只在加载阶段，FMA 只在计算阶段，两者根本不同时竞争发射槽
//
//     真正的机制：4条 scalar store > 1条 vector store → 加载阶段耗时增加
//       → __syncthreads 屏障更晚解除 → 计算阶段推迟启动 → FMA 等待
//
//       kernel_6：[──加载──]──sync──[────FMA────]
//       kernel_7：[────加载────]──sync──[────FMA────]
//                      ↑ 多出这段时间内 FMA 等待
//
//     SM 上有其他 block 时可部分缓解（其他 block 的 warp 在跑 FMA），
//     但本 block 的计算时间占比下降，整体吞吐仍降低
//
//   ③ Bank conflict 是 latency 问题（可被隐藏），指令数是 throughput 问题（不可隐藏）
//
//     Bank conflict 的隐藏机制：
//       某 warp 遇到 bank conflict → stall → 调度器切换到其他 ready warp
//       FMA 单元继续执行其他 warp → bank conflict 延迟被掩盖
//       compute-bound 时有大量 ready warp → 调度器从不缺活 → conflict 影响极小
//
//     指令数增加无法隐藏：
//       每条 store 指令延长加载阶段，sync 屏障推迟，FMA 启动推迟
//       这是硬性时序开销，无论调度器多优秀都无法绕过
//
//   ④ 结论：
//     读 conflict（4-way）：latency 类型，compute-bound 时被 warp 切换掩盖，影响有限
//     写指令数（×4）      ：延长加载阶段，FMA 等待 sync，无法掩盖
//     净效果：加载阶段开销 > 读 conflict 消除收益 → kernel_7 略慢
//
//   ┌──────────────┬─────────────────────┬──────────────────────┬──────────────┐
//   │              │ Bs 写入             │ Bs 读 conflict       │ 实测         │
//   ├──────────────┼─────────────────────┼──────────────────────┼──────────────┤
//   │ kernel_6     │ 1条 v4 store，4-way │ 4-way（每 dotIdx）   │ 4699 GFLOPS  │
//   │ kernel_7     │ 4条 scalar store，  │ 无                   │ 4377 GFLOPS  │
//   │              │ 2-way               │                      │              │
//   └──────────────┴─────────────────────┴──────────────────────┴──────────────┘
//
//   正确消除 Bs bank conflict 的方法是 kernel_8 的 padding 方案：
//   保留 float4 向量写入，仅在 Bs 末尾追加 padding 列使行步长不是 32 的整数倍
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
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <float.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN),1) gemmResolveBankExtraCol(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  const int extraCols = 5;
  __shared__ float Bs[BK * (BN + extraCols)];

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

    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmp.y;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmp.z;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmp.w;
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
        regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
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
__global__ void __launch_bounds__((BM * BN) / (TM * TN),1) gemmResolveBankExtraCol_v2(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C) {
  // 该block负责计算的C矩阵元素的起始行数和列数
  const uint initRow {blockIdx.y * BM};
  const uint initCol {blockIdx.x * BN};



  // 向量加载
  // 加载As
  static_assert( BK % 4 == 0 && "列组不为整数");
  assert( blockDim.x % (BK / 4)  == 0 && "所有列不能覆盖");
  assert(BK * BM % (blockDim.x * 4) == 0 && "不能覆盖所有行");
  const uint innerColGroupAs { threadIdx.x % (BK / 4) };
  const uint innerRowAs { threadIdx.x / (BK / 4) };
  const uint innerRowNumAs { blockDim.x /  (BK / 4) };
  // 加载As
  static_assert( BN % 4 == 0 && "列组不为整数");
  assert( blockDim.x % (BN / 4)  == 0 && "所有列不能覆盖");
  assert(BK * BN % (blockDim.x * 4) == 0 && "不能覆盖所有行");
  const uint innerColGroupBs { threadIdx.x % (BN / 4) };
  const uint innerRowBs { threadIdx.x / (BN / 4) };
  const uint innerRowNumBs { blockDim.x /  (BN / 4) };
  constexpr uint extraCols {4};

  // 存储该block需要加载的BM * BN个元素的乘积，每个轮次取BK列
  __shared__ float As[BM * BK];

  // 行主序
  // __shared__ float Bs[BK * BN];
  // PADDING
  __shared__ float Bs[BK * (BN + extraCols)];


  // 计算
  static_assert((BN %  TN) == 0 && "列组不是整数");
  assert(blockDim.x % (BN / TN) == 0 && "不能覆盖完整列组");
  assert(BK * BM % (blockDim.x * TM * TN) == 0 && "不能覆盖完整行组");
  const uint threadRowGroup {threadIdx.x / (BN/ TN)};
  const uint threadColGroup {threadIdx.x % (BN / TN)};
  float threadResults[TM * TN] {0.0f};
  float tmpAs[TM];
  float tmpBs[TN];

  // 写回C
  static_assert((TN % 4) == 0 && "列组不是整数");
  const uint writeBackColGroupNum {TN / 4};




  // K方向循环
  for (uint outterColIdx {}; outterColIdx < K; outterColIdx += BK) {
    //加载As
    for (uint rowIdx {}; rowIdx < BM; rowIdx+=innerRowNumAs) {
      float4 vecAs = reinterpret_cast<float4 *>(&A[(initRow + innerRowAs + rowIdx) * K + outterColIdx + innerColGroupAs * 4])[0];
      // 行主序
      // As[(innerRowAs + rowIdx) * BK + innerColGroupAs * 4] = vecAs.x;
      // As[(innerRowAs + rowIdx ) * BK + innerColGroupAs * 4 + 1] = vecAs.y;
      // As[(innerRowAs + rowIdx) * BK + innerColGroupAs * 4 + 2] = vecAs.z;
      // As[(innerRowAs + rowIdx) * BK + innerColGroupAs * 4 + 3] = vecAs.w;

      // 列主序
      As[(innerColGroupAs * 4) * BM + innerRowAs + rowIdx] = vecAs.x;
      As[(innerColGroupAs * 4+ 1) * BM + innerRowAs + rowIdx] = vecAs.y;
      As[(innerColGroupAs * 4+ 2) * BM + innerRowAs + rowIdx] = vecAs.z;
      As[(innerColGroupAs * 4+ 3) * BM + innerRowAs + rowIdx] = vecAs.w;
    }

    // 加载Bs
    for (uint rowIdx {}; rowIdx < BK; rowIdx+=innerRowNumBs) {
      float4 vecBs = reinterpret_cast<float4 *>(&B[(innerRowBs + rowIdx + outterColIdx) * N + initCol + innerColGroupBs * 4])[0];
      // 行主序
      // BN / 4 = 128 / 4 = 32，
      // 一个WARP32个线程，innerColGroupBs=0，1，2，...，31，innerRowBs=0，0，0，0，，0，0
      // rowIdx固定，sts32合并为sts128，再phase后，前8个线程访问Bs的一行，无bank conflict
      // Bs[(innerRowBs +rowIdx) * BN + innerColGroupBs * 4] = vecBs.x;
      // Bs[(innerRowBs +rowIdx) * BN + innerColGroupBs * 4 + 1] = vecBs.y;
      // Bs[(innerRowBs +rowIdx) * BN + innerColGroupBs * 4 + 2] = vecBs.z;
      // Bs[(innerRowBs +rowIdx) * BN + innerColGroupBs * 4 + 3] = vecBs.w;

      // PADDING后
      // 一个WARP32个线程，innerColGroupBs=0，1，2，...，31，innerRowBs=0，0，0，0，，0，0
      //rowIdx固定，sts32合并为sts128，再phase后，前0-7个线程访问Bs的一行，以此类推
      // 下一个warp32个线程，innerColGroupBs=0，1，2，...，31，innerRowBs=0，0，0，0，，0，0
      //rowIdx固定，sts32合并为sts128，再phase后，前0-7个线程访问Bs的一行，以此类推
      //因为现在列数为BN + extraCols，所以在0-7个线程中间会空出extraCols个bank，
      //有没有空出extraCols个bank不会影响bank conflict，但是合并指令sts128要求16字节对齐，一个bank是4字节，所以空出的extraCols需要是4的倍数，
      // 如果padding是5，退化为 4×STS.32，出现了：4×指令数和写入 4-way bank conflict的情况，和kernel7一样，
      // 并且由于padding并没有解决计算是Bs的bank conflict，导致性能比kernel7的还要差。为Average elapsed time: (0.036606) s, performance: (3754.5) GFLOPS. siz
      Bs[(innerRowBs + rowIdx) * (BN + extraCols) + innerColGroupBs * 4] = vecBs.x;
      Bs[(innerRowBs + rowIdx) * (BN + extraCols) + innerColGroupBs * 4 + 1] = vecBs.y;
      Bs[(innerRowBs + rowIdx) * (BN + extraCols) + innerColGroupBs * 4 + 2] = vecBs.z;
      Bs[(innerRowBs + rowIdx) * (BN + extraCols) + innerColGroupBs * 4 + 3] = vecBs.w;

    }

    __syncthreads();

    // 计算结果
    for (uint innerColIdx {}; innerColIdx < BK; ++innerColIdx) {
      for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
        // 行主序
        // tmpAs[rowIdx] = As[(threadRowGroup * TM + rowIdx) * BK + innerColIdx];

        // 列主序
        tmpAs[rowIdx] = As[innerColIdx * BM + threadRowGroup * TM + rowIdx];
      }

      for (uint colIdx {}; colIdx < TN; ++colIdx) {
        // 行主序
        // BN / TN=128/8=16，
        // 一个warp32个线程，每个线程读取的元素相差8个，同一行的0，1，...，15，0，1，...，15，
        // word=16， bank = 32 / 8 = 4，bank conflict=16 / 4 = 4-way bank conflict
        //
        // ── 详细推导 ────────────────────────────────────────────────────────────
        // threadColGroup = threadIdx.x % (BN/TN) = threadIdx.x % 16
        // warp 内取值：0,1,...,15,0,1,...,15（每值 2 个线程，访问同一地址 → broadcast）
        //
        // 对固定 innerColIdx 和 colIdx：
        //   地址 = innerColIdx * BN + colIdx + threadColGroup * TN
        //   Bank = (colIdx + threadColGroup * 8) % 32  （BN=128≡0 mod 32，行首不影响 bank）
        //
        //   threadColGroup=0  → bank = (colIdx) % 32
        //   threadColGroup=4  → bank = (colIdx + 32) % 32 = colIdx % 32  ← 与 0 同 bank！
        //   threadColGroup=8  → 同 bank；threadColGroup=12 → 同 bank
        //
        // 步长 = TN = 8 banks，周期 = 32 / 8 = 4
        // 唯一 threadColGroup 数 = BN/TN = 16，每 bank 命中 16/4 = 4 个不同地址
        // → 4-way bank conflict（上方 "word=16" 指唯一 threadColGroup 数，非数据字宽）
        // tmpBs[colIdx] = Bs[innerColIdx * BN + colIdx + threadColGroup * TN];

        // padding
        // BN / TN=128/8=16，
        // 一个warp32个线程，每个线程读取的元素相差8个，同一行的0，1，...，15，0，1，...，15，
        // 步长 = TN = 8 banks，周期 = 32 / 8 = 4，每 bank 命中 16/4 = 4 个不同地址
        // 由于innerColIdx不变，padding并没有改善读Bs的bank conflict。
        // padding 只改变行首 bank（加法常数），不改变两个 threadCol 之间的相对差值。thread之间的步长是8，这个差值与 padding 无关，4-way conflict 不可能通过 padding 消除。
        tmpBs[colIdx] = Bs[innerColIdx * (BN + extraCols) + colIdx + threadColGroup * TN];

      }

      for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
        for (uint colIdx {}; colIdx < TN; ++colIdx) {
          threadResults[rowIdx * TN + colIdx] += tmpAs[rowIdx] * tmpBs[colIdx];
        }
      }
    }

    __syncthreads();
  }

  // 向量写回 C
  // 该线程负责 TM 行 × TN 列的结果，TN 列按每组 4 个 float 划分为 writeBackColGroupNum 组
  // colIdx ∈ [0, TN/4)，每次用 float4 向量写回 4 个连续列，共写 TM × (TN/4) 次
  //
  // C 的全局地址：
  //   行：initRow + threadRowGroup * TM + rowIdx   （该线程负责的第 rowIdx 行）
  //   列：initCol + threadColGroup * TN + colIdx*4  （该线程负责的第 colIdx 个 float4 组起始列）
  //
  // threadResults 线性索引：rowIdx * TN + colIdx * 4（与 colIdx*4 对齐，4 个连续元素）
  //
  // 等价于原始写法（resIdxN += 4）：
  //   for (resIdxN = 0; resIdxN < TN; resIdxN += 4)
  //   colIdx * 4 == resIdxN，逐步覆盖 TN 列
  for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
    for (uint colIdx {}; colIdx < writeBackColGroupNum; ++colIdx) {
      // 先读出 C 的旧值，再执行 alpha*A*B + beta*C 的线性组合后写回
      float4 vecWriteBack { reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroup * TN + colIdx * 4])[0] };
      vecWriteBack.x = alpha * threadResults[rowIdx * TN + colIdx * 4    ] + beta * vecWriteBack.x;
      vecWriteBack.y = alpha * threadResults[rowIdx * TN + colIdx * 4 + 1] + beta * vecWriteBack.y;
      vecWriteBack.z = alpha * threadResults[rowIdx * TN + colIdx * 4 + 2] + beta * vecWriteBack.z;
      vecWriteBack.w = alpha * threadResults[rowIdx * TN + colIdx * 4 + 3] + beta * vecWriteBack.w;
      reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroup * TN + colIdx * 4])[0] = vecWriteBack;
    }
  }
}
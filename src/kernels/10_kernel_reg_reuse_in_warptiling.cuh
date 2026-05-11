#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
constexpr uint K10_NUM_THREADS = 128;
// CUDA 内置的 warpSize 本质是 extern const int warpSize：
//   extern 只是一个"承诺"：该符号存在，定义在别处，编译器先放行。
//   何时兑现取决于机制：普通 C++ 全局变量由链接器在链接期解析；
//   而 warpSize 类似 blockDim/threadIdx，由 kernel 执行时 GPU 硬件上下文提供，链接期同样未知。
//   因此无法用于 constexpr 表达式和模板参数计算，故手动定义为编译期常量。
constexpr  int WARPSIZE = 32;

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    gemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;



  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

// ── kernel 10 相比 kernel 9 的两点核心改进（相互独立）────────────────────────────
//
// 【改进1：主动将 WM 设为小于 BM，每个 warp 聚焦更小的 SMEM 区域】
//   kernel 9：WM 与 BM 在代码上已解耦（WM = threadRowNum×TM，BM 独立），
//     但自动调优后参数恰好使 WM = BM，block 内只有 1 个 warp tile
//   kernel 10：主动选择 WM < BM（设计决策，非参数巧合）
//     → block 内有 BM/WM × BN/WN 个 warp tile，每个 warp 聚焦更小的连续 SMEM 区域，
//       改善 warp 级访问局部性（与改进2 相互独立，即使 WM = BM 改进2 也可生效）
//
// 【改进2：WM 与线程布局解耦，使 threadRowIterNum > 1，实现寄存器复用】
//   关键公式：
//     threadrowNumPerIter = K10_NUM_THREADS / (WN/TN) * TM   ← 仅取决于线程数和 TM/TN，与 WM 无关
//     threadRowIterNum    = WM / threadrowNumPerIter          ← WM 越大，复用次数越多
//
//   kernel 9：WM = threadRowNum×TM，WM 与 BM 已解耦（参数恰好使 WM = BM 只是结果），
//     但 WM 由线程布局推导，恒等于 threadrowNumPerIter：
//       WM = threadRowNum×TM = (NUM_THREADS / threadColNum)×TM
//          = (NUM_THREADS / (WN/TN))×TM = threadrowNumPerIter
//     → threadRowIterNum = 1 是构造上的必然，与具体参数无关，无寄存器复用
//
//   kernel 10 的修正：WM 作为完全自由的模板参数，不再由线程布局推导，
//     可设置 WM > threadrowNumPerIter
//     → threadRowIterNum > 1，每次 innerCol 一次性预加载 threadRowIterNum×TM 行，
//       regAsCache[rowIdx] 在 colIdx 循环中复用 TN 次，regBsCache[colIdx] 复用 threadRowIterNum×TM 次
//
//   gemmWarptiling_v2 的循环结构：
//     for wintRow in [0, BM, WM):
//       for wintCol in [0, BN, WN):
//         for innerCol in [0, BK):
//           load regAsCache[threadRowIterNum*TM]  ← 一次预加载该线程在此 warp tile 内的所有行
//           load regBsCache[TN]                   ← 预加载该线程负责的 TN 列
//           for rowIdx in [0, threadRowIterNum*TM):
//             for colIdx in [0, TN):
//               threadResult[...] += regAsCache[rowIdx] * regBsCache[colIdx]
// ────────────────────────────────────────────────────────────────────────────
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int TM, const int TN>
__global__ void __launch_bounds__(K10_NUM_THREADS,1) gemmWarptiling_v2(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {

  // 该block负责计算的起始行和列
  const uint initRow {blockIdx.y * BM};
  const uint initCol {blockIdx.x * BN};

  // 该block的SMEM静态存储
  __shared__ float As[BM * BK];
  // 加padding
  constexpr uint extraCol { 4 };
  __shared__ float Bs[BK * (BN + extraCol)];

  // 加载阶段
  // 该线程负责加载的As和Bs的行组和列
  // As
  static_assert(BK % 4 == 0 && "向量加载As的列组不是整数");
  // 暗含了K12_NUM_THREADS不能小于(BK / 4)
  static_assert(K10_NUM_THREADS % (BK / 4) == 0 && "block不能覆盖完整的列组");
  static_assert(BM * BK % (K10_NUM_THREADS * 4) == 0 && "block不能覆盖完整的行");
  // 两者等价
  // % 和 / 优先级相同，从左到右结合，实际被解析为 (BM % K10_NUM_THREADS) / (BK/4) == 0，不是预期的 BM % (K10_NUM_THREADS / (BK/4)) == 0，需加括号。
  static_assert(BM % (K10_NUM_THREADS / (BK / 4)) == 0 && "block在n次迭代中不能覆盖完整的行");
  const uint rowNumAsPerIter {K10_NUM_THREADS / (BK / 4)};
  const uint loadAsRow {threadIdx.x / (BK / 4)};
  const uint loadAsColGroup {threadIdx.x % (BK / 4)};
  //Bs
  static_assert(BN % 4 == 0 && "向量加载的Bs列组不是整数");
  // 暗含了K12_NUM_THREADS不能小于(BK / 4)
  static_assert(K10_NUM_THREADS % (BN / 4) == 0 && "block不能覆盖完整的列组");
  static_assert(BN * BK % (K10_NUM_THREADS * 4) == 0 && "block不能覆盖完整的行");
  // 两者等价
  static_assert(BK % (K10_NUM_THREADS / (BN / 4)) == 0 && "block在n次迭代中不能覆盖完整的行");
  const uint loadBsRow {threadIdx.x / (BN / 4)};
  const uint loadABsColGroup {threadIdx.x % (BN / 4)};
  const uint rowNumBsPerIter {K10_NUM_THREADS / (BN / 4)};

  // 计算阶段
  // 暗含BM要大于WM
  static_assert(BM % WM == 0 && "经过n次迭代，能覆盖完整的BM");
  // 暗含BN要大于WN
  static_assert(BN % WN == 0 && "经过n次迭代，能覆盖完整的BM");
  constexpr  uint BRowIterNum{ BM / WM };
  constexpr  uint BColNIterNum{ BN / WN };
  // 暗含WN大于TN
  static_assert(WN % TN == 0 && "列组不是整数");
  // 暗含WM大于TM
  static_assert(WM % TM == 0 && "行组不是整数");
  static_assert(K10_NUM_THREADS % (WN / TN) == 0 && "block不能覆盖完整列组");
  // 暗含WM要大于K10_NUM_THREADS / (WN / TN) * TM，即大于threadrowNumPerIter，threadRowIterNum永远不会是0
  static_assert(WM % (K10_NUM_THREADS / (WN / TN) * TM) == 0 && "block在n次迭代中不能覆盖完整行组，需要处理边界情况");
  // 等价于
  static_assert((WM * WN) % (K10_NUM_THREADS * TM * TN) == 0);
  const uint threadColGroup {threadIdx.x % (WN / TN)};
  const uint threadrowNumPerIter {K10_NUM_THREADS / (WN / TN) * TM};
  const uint threadRowIterNum {WM / threadrowNumPerIter};
  const uint threadRowGroup {threadIdx.x / (WN / TN)};
  // 寄存器复用：regAsCache 一次预加载该线程在一个 warp tile 内的所有行，
  // 每个元素在内层 colIdx 循环中复用 TN 次；regBsCache 同理复用 threadRowIterNum*TM 次。
  float regAsCache[threadRowIterNum * TM];
  float regBsCache[TN];
  // threadResult 逻辑布局（4D 行主序）：[BRowIterNum][threadRowIterNum*TM][BColNIterNum][TN]
  //   BRowIterNum  = BM/WM（block 内 warp tile 行数），BColNIterNum = BN/WN（warp tile 列数）
  //   threadRowIterNum*TM：该线程在一个 warp tile 内负责的行数，TN：负责的列数
  // 展开索引：(wintRow/WM * threadRowIterNum*TM + rowIdx) * BColNIterNum*TN
  //           + wintCol/WN * TN + colIdx
  //   wintRow/WM 次迭代时行跨步为 threadRowIterNum*TM，wintCol/WN 次迭代时列跨步为 TN
  float threadResult[BRowIterNum * threadRowIterNum * TM * BColNIterNum * TN] {0.0f};

  // 写回C
  static_assert(TN % 4 == 0 && "列组不是整数");


  // 沿K方向
  for (uint outterIdx {}; outterIdx < K; outterIdx+=BK) {
    // 加载As
    for (uint row {}; row < BM; row+=rowNumAsPerIter) {
      float4 loadVecAs { reinterpret_cast<float4 *>(&A[(initRow + loadAsRow + row) * K + loadAsColGroup * 4 + outterIdx])[0] };
      // 行主序
      // As[(loadAsRow + row) * BK + loadAsColGroup * 4] = loadVecAs.x;
      // As[(loadAsRow + row) * BK + loadAsColGroup * 4 + 1] = loadVecAs.y;
      // As[(loadAsRow + row) * BK + loadAsColGroup * 4 + 2] = loadVecAs.z;
      // As[(loadAsRow + row) * BK + loadAsColGroup * 4 + 3] = loadVecAs.w;

      // 列主序
      As[(loadAsColGroup * 4) * BM + loadAsRow + row] = loadVecAs.x;
      As[(loadAsColGroup * 4 + 1) * BM + loadAsRow + row] = loadVecAs.y;
      As[(loadAsColGroup * 4 + 2) * BM + loadAsRow + row] = loadVecAs.z;
      As[(loadAsColGroup * 4 + 3) * BM + loadAsRow + row] = loadVecAs.w;
    }

    // 加载Bs
    for (uint row {}; row < BK; row+=rowNumBsPerIter) {
      float4 loadVecBs { reinterpret_cast<float4 *>(&B[(loadBsRow + outterIdx + row) * N + initCol + loadABsColGroup * 4])[0] };
      // 行主序
      // Bs[(loadBsRow + row) * BN + loadABsColGroup * 4] = loadVecBs.x;
      // Bs[(loadBsRow + row) * BN + loadABsColGroup * 4 + 1] = loadVecBs.y;
      // Bs[(loadBsRow + row) * BN + loadABsColGroup * 4 + 2] = loadVecBs.z;
      // Bs[(loadBsRow + row) * BN + loadABsColGroup * 4 + 3] = loadVecBs.w;

      // 行主序 + padding
      Bs[(loadBsRow + row) * (BN + extraCol) + loadABsColGroup * 4] = loadVecBs.x;
      Bs[(loadBsRow + row) * (BN + extraCol) + loadABsColGroup * 4 + 1] = loadVecBs.y;
      Bs[(loadBsRow + row) * (BN + extraCol) + loadABsColGroup * 4 + 2] = loadVecBs.z;
      Bs[(loadBsRow + row) * (BN + extraCol) + loadABsColGroup * 4 + 3] = loadVecBs.w;
    }

    __syncthreads();

    // 计算阶段
    for (uint wintRow{}; wintRow < BM; wintRow+=WM) {
      for (uint wintCol{}; wintCol < BN; wintCol+=WN) {
        for (uint innerCol {}; innerCol < BK; ++innerCol) {
          for (uint rowIdx{}; rowIdx < threadRowIterNum * TM; ++rowIdx) {
            // 行主序
            // regAsCache[rowIdx] = As[(wintRow + rowIdx + threadRowGroup * TM * threadRowIterNum) * BK + innerCol];
            // 列主序
            regAsCache[rowIdx] = As[innerCol * BM + wintRow + rowIdx + threadRowGroup * TM * threadRowIterNum];
          }
          for (uint colIdx {}; colIdx < TN; ++colIdx) {
            // 行主序
            // regBsCache[colIdx] = Bs[innerCol * BN + wintCol + colIdx + threadColGroup * TN];
            // 列主序 + padding
            regBsCache[colIdx] = Bs[innerCol * (BN + extraCol) + wintCol + colIdx + threadColGroup * TN];
          }

          for (uint rowIdx{}; rowIdx < threadRowIterNum * TM; ++rowIdx) {
            for (uint colIdx {}; colIdx < TN; ++colIdx) {
              // wintRow/WM次迭代的时候，threadResult行跨步是threadRowIterNum * TM，wintCol/WN次迭代的时候，threadResult跨步是TN
              threadResult[(wintRow/WM * threadRowIterNum * TM + rowIdx) * BColNIterNum * TN + colIdx + wintCol/WN * TN] += regAsCache[rowIdx] * regBsCache[colIdx];
            }
          }
        }
      }
    }

    __syncthreads();
  }

  // 写回C
  // 易错：必须有 rowIdx 循环 [0, threadRowIterNum*TM)，每个线程在一个 warp tile 内负责多行；
  //       必须有 colIdx 循环 [0, TN/4)，用 float4 向量写，每次覆盖4列；
  //       两层循环缺一则只写回第一行/第一组列，其余结果丢失。
  for (uint wintRow{}; wintRow < BM; wintRow+=WM) {
    for (uint wintCol{}; wintCol < BN; wintCol+=WN) {
      for (uint rowIdx{}; rowIdx < threadRowIterNum * TM; ++rowIdx) {
        for (uint colIdx {}; colIdx < TN/4; ++colIdx) {
          // block的起始行initRow,起始列initCol
          // 线程负责的BM中的位置:threadRowGroup * threadRowIterNum * TM + wintRow
          // 线程负责的BN中的位置： + threadColGroup * (TN / 4) * 4 + wintCol
          float4 writeBackToCVec { reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * threadRowIterNum * TM + wintRow + rowIdx) * N + initCol + threadColGroup * TN + wintCol + colIdx * 4])[0] };
          // 线程存储的BM对应的寄存器位置：wintRow/WM  * threadRowIterNum * TM
          // 线程存储的BN对应的寄存器位置：wintRow/WM  * (wintCol / WN) *  (TN / 4) * 4
          writeBackToCVec.x = alpha * threadResult[(wintRow/WM  * threadRowIterNum * TM + rowIdx) * BColNIterNum * TN + (wintCol / WN) * TN + colIdx * 4] + beta * writeBackToCVec.x;
          writeBackToCVec.y = alpha * threadResult[(wintRow/WM  * threadRowIterNum * TM + rowIdx) * BColNIterNum * TN + (wintCol / WN) * TN +  colIdx * 4 + 1] + beta * writeBackToCVec.y;
          writeBackToCVec.z = alpha * threadResult[(wintRow/WM  * threadRowIterNum * TM + rowIdx) * BColNIterNum * TN + (wintCol / WN) * TN +  colIdx * 4 + 2] + beta * writeBackToCVec.z;
          writeBackToCVec.w = alpha * threadResult[(wintRow/WM  * threadRowIterNum * TM + rowIdx) * BColNIterNum * TN + (wintCol / WN) * TN +  colIdx * 4 + 3] + beta * writeBackToCVec.w;
          reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * threadRowIterNum * TM + wintRow + rowIdx) * N + initCol + threadColGroup * TN + wintCol + colIdx * 4])[0] = writeBackToCVec;
        }
      }
    }
  }

}


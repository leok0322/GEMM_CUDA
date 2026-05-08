#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 为什么定义在此头文件而非 runner.cu：
//   __launch_bounds__ 必须写在 __global__ 函数定义处，是给编译器的寄存器分配提示，
//   而非运行时参数；rowStrideA/B 也是 constexpr，须在 kernel 作用域内可见。
//   两者都要求该常量在本 .cuh 文件的作用域内，无法移到 runner.cu。
//
// 为什么固定 256、不参与自动搜索：
//   代码里真正硬编码的原始常量是下方的 WM=TM*16 / WN=TN*16 里的 "16"，
//   256 是由它推导出来的结果，而非输入：
//     线程排列：16（M方向）× 16（N方向）= 256
//     NUM_THREADS = (WM/TM) × (WN/TN) = 16 × 16 = 256
//   无论 BM/BN/TM/TN 取何值该式恒为 256，搜索其他值会导致输出覆盖缺失或重复。
//
//   设计动机：256 是 GPU 调优中的经验性好值（8个warp，足够隐藏延迟且寄存器压力
//   不过高），先选定 256，再反推用 16×16 正方形排列映射线程，WM=TM*16 随之确定。
//   即：设计时先拍定 256，代码中的原始常量却是 16。
constexpr int K9_NUM_THREADS = 16 * 16;

// ── CPU/GPU 执行模型与断言机制 ───────────────────────────────────────────────
// __global__ kernel 运行在 GPU 上，不是 CPU：
//   CPU（host）调用 kernel<<<grid, block>>>() 后立即返回（异步发射），
//   GPU（device）并发执行 kernel 函数体，CPU 与 GPU 各跑各的时间线：
//
//   CPU 时间线: ──launch──────────────────────────────→ 继续执行
//   GPU 时间线:            ──────────────kernel──────→ 完成
//
// cudaDeviceSynchronize() 是两条时间线之间的屏障（barrier）：
//   CPU 在此处阻塞，直到 GPU 上所有已提交操作全部完成。
//   非 Async 版本的 cudaMemcpy 也隐式同步：CPU 等 GPU 完成后才开始拷贝，
//   因此 gemm.cu 中将结果拷回 host 的 cudaMemcpy 已起到同步作用，
//   不需要再额外调用 cudaDeviceSynchronize()。
//
// static_assert 与 assert 在 kernel 中的区别：
//   static_assert：纯编译期，nvcc 实例化模板时求值，不产生任何 GPU 指令，无限制。
//                  Debug 和 Release 模式下均有效，与 NDEBUG 无关。
//   assert       ：运行期，GPU 线程触发时将错误信息写入预留的 GPU 内存缓冲区，
//                  kernel 终止；CPU 侧在下一个同步点（cudaDeviceSynchronize /
//                  cudaMemcpy 等）读取缓冲区并将错误打印到 stderr。
//                  每个触发 assert 的 GPU 线程各自独立打印，可能出现几百条。
//                  编译时定义 NDEBUG（Release 模式自动定义）→ assert 被预处理器
//                  展开为 ((void)0)，完全移除，越界访问静默发生，得到错误结果而不报错。
//
// Release 模式下的校验方案：
//   编译期值（模板参数 BM/BN/BK/TM/TN）：用 static_assert，release和debug两种模式均有效 ✓
//   运行期值（用户传入的 M/N/K）        ：assert 在 Release 下被移除，
//     只能用显式 if 手动检查，建议在 CPU 侧 launch 前做，比在 GPU kernel 内检查更高效：
//       if (M % K9_BM != 0 || N % K9_BN != 0) {
//           fprintf(stderr, "M/N must be multiples of BM/BN\n");
//           exit(EXIT_FAILURE);
//       }
//     fprintf(stderr) 与 NDEBUG 无关，Debug 和 Release 下均正常打印。
//
// assert 与 cudaGetLastError() 的 stderr 打印路径对比：
//
//   【assert】GPU 线程触发 → 直接写入 stderr：
//     GPU 线程 assert 失败
//       → CUDA runtime 将断言信息（文件/行号/block/thread 坐标）写入 GPU 缓冲区
//       → 同步点（cudaDeviceSynchronize / cudaMemcpy）触发
//       → CUDA runtime 在 CPU 侧调用 fprintf(stderr, ...) 直接写到 stderr
//     assert 消息本身不经过任何用户代码，由 CUDA runtime 内部打印。
//
//   【cudaGetLastError()】错误码 → cudaCheck → fprintf(stderr)：
//     kernel 执行出错（如参数非法、资源不足）
//       → CUDA runtime 将错误码存入当前线程的错误寄存器
//       → cudaGetLastError() 读取并清除该错误寄存器，返回 cudaError_t（整数枚举）
//          此函数本身不打印任何内容，只是返回一个错误码
//       → cudaCheck(cudaError_t, file, line) 收到错误码
//       → 若 != cudaSuccess，调用 fprintf(stderr, "[CUDA ERROR]...") 写到 stderr
//     打印路径：错误码（整数）→ cudaCheck → fprintf(stderr) → stderr
//
//   两者在脚本中的去向（run_all_kernels.sh 中 2>&1 | tee -a error.txt）：
//     stderr（fd 2）被 2>&1 合并到 stdout（fd 1，管道写端），
//     tee 从管道读取后同时写到终端和 error.txt，两类错误信息最终都落入同一文件。
//
// assert 错误消息 与 cudaCheck(cudaDeviceSynchronize()) 拿到的信息的差异：
//
//   assert 触发后，终端上会出现两条独立的错误信息：
//
//   第 1 条：CUDA runtime 刷新 GPU 缓冲区时直接打印，内容详细：
//     kernel.cu:42: gemmAutotuned: block: [2,0,0], thread: [127,0,0]
//     Assertion `row < M` failed.
//     包含：源文件名、行号、kernel 函数名、block 坐标、thread 坐标、断言条件。
//     由 CUDA runtime 内部生成，不经过任何用户代码。
//
//   第 2 条：cudaDeviceSynchronize() 因 assert 失败返回 cudaErrorAssert，
//     cudaCheck 调用 cudaGetErrorString(cudaErrorAssert) 得到固定通用字符串：
//       "device-side assert triggered"
//     再通过 fprintf(stderr, "[CUDA ERROR] at file %s:%d:\n%s\n", ...) 打印：
//       [CUDA ERROR] at file gemm.cu:163:
//       device-side assert triggered
//     cudaErrorAssert 只是一个整数枚举值，不携带具体断言信息；
//     cudaGetErrorString 只能将枚举翻译为固定字符串，无法还原原始断言内容。
//
//   调试时有用的是第 1 条（具体位置和条件），第 2 条只是兜底保障（确保程序不在
//   错误状态下继续运行，cudaCheck 在此处调用 exit(EXIT_FAILURE) 终止进程）。
//
// 模板非类型参数天然是编译期常量，无需任何修饰符：
//   template <int BM>         ✓ 标准写法
//   template <const int BM>   ✓ 合法但 const 冗余（标准规定非类型参数隐式 const）
//   template <constexpr int BM> ✗ 语法错误，constexpr 不是模板参数列表的合法关键字
// 此处 const 是冗余写法，与 template <int BM, int BN, ...> 完全等价。
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    gemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // ── 与前面 kernel（如 kernel 6）的整除逻辑差异 ──────────────────────────────
  // kernel 6：threadCol = threadIdx % (BN/TN)，范围 [0, BN/TN) 随参数动态变化。
  //           blockDim = BM*BN/(TM*TN) 由 BM/BN/TM/TN 推导，BN 决定 threadCol 范围。
  //           约束：BN%TN==0，blockDim%(BN/TN)==0（先有 BN/BM，再推 blockDim）。
  //
  // kernel 9：先固定 blockDim=256=16×16，硬编码 WM=TM*16 / WN=TN*16，
  //           threadCol = threadIdx % (WN/TN) = threadIdx % 16，范围固定为 [0,16)，与 BN 无关。
  //           约束变为：BN%(TN*16)==0（先有 blockDim，倒推 BN 必须是 WN 的整数倍）。
  //           两套约束不等价：反例 BN=64,TN=8 满足 kernel 6 的约束，
  //           但 kernel 9 中 threadCol=15 访问第 127 列，而 Bs 只有 64 列，越界。
  // 此处必须用 constexpr 而非 const：
  //   TM/TN 是模板参数（编译期值），WM/WN 要用于 WMITER/WNITER 等需要编译期值的场合。
  //   constexpr 显式声明并让编译器强制验证；const 在字面量初始化时碰巧也是编译期常量，
  //   但无法保证——runner.cu 中 K9_BK = 16 用 const 是 C 风格习惯写法，功能等价但意图不明确。
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  // WMITER/WNITER：block tile 在 M/N 方向被分成多少个 warp tile。
  // runner.cu 的 static_assert 已保证 BM%WM==0、BN%WN==0，CEIL_DIV 等价于整除。
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // threadCol 范围固定为 [0, WN/TN) = [0, 16)，由 WN 决定，与 BN 无关。
  // kernel 6 的 threadCol 范围是 [0, BN/TN)，随 BN 动态变化——这是两者的核心差异。
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

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
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * WNITER * TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // block into registers
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();
    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
  }

  // write out the results
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                         resIdxN])[0];
          // perform GEMM update in reg
          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
              tmp;
        }
      }
    }
  }
}

template <const int BM,const int BN, const int BK, const int TM,const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS,1) gemmAutotuned_v2 (float* A, float* B, float* C,int M, int N, int K, float alpha, float beta) {

  // 该block负责计算的起始行和列
  const uint initRow {blockIdx.y * BM};
  const uint initCol {blockIdx.x * BN};

  // 该block的静态SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 该线程负责加载的行和列
  // 只要线程0进行静态断言
  // 可以在runner.cu中进行断言，不要在kernel中做静态或者动态断言
  // if (threadIdx.x == 0) {
  //   static_assert(BK % 4 == 0 && "列组不是整数");
  //   static_assert(K9_NUM_THREADS % (BK / 4) == 0 && "不能覆盖完整列组");
  //   static_assert(BM * BK % (K9_NUM_THREADS * 4) == 0 && "不能覆盖完整行");
  // }
  const uint loadAsColGroup = threadIdx.x % (BK / 4) ;
  const uint  loadAsRowGroup = threadIdx.x / (BK / 4) ;
  // 只要线程0进行静态断言
  // 可以在runner.cu中进行断言，不要在kernel中做静态或者动态断言
  // if (threadIdx.x == 0) {
  //   static_assert(BN % 4 == 0 && "列组不是整数");
  //   static_assert(K9_NUM_THREADS % (BN / 4) == 0 && "不能覆盖完整列组");
  //   static_assert(BN * BK % (K9_NUM_THREADS * 4) == 0 && "不能覆盖完整行");
  // }
  const uint loadBsColGroup = threadIdx.x % (BN / 4);
  const uint  loadBsRowGroup = threadIdx.x / (BN / 4);

  // 该线程负责计算的行和列
  const uint WN { 16 * TN};
  const uint WM {16 * TM};
  static_assert(BM % WM == 0 && "能覆盖完整的行");
  static_assert(BN % WN == 0 && "能覆盖完整的列");

  // 沿k方向循环
  for (uint outerColidx {}; outerColidx < K; outerColidx+=BK) {
    // 加载As和Bs


  }



}
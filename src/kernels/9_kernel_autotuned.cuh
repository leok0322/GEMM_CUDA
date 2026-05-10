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
constexpr uint K9_NUM_THREADS = 16 * 16;

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

// ── kernel 9 (v2) 相比 kernel 8 性能提升的原因分析 ──────────────────────────
  //
  // 两个 kernel 的 shared memory 布局完全相同：
  //   As：列主序（转置存储），读 As 时 BM 方向连续 → 消除 bank conflict
  //   Bs：行主序 + extra col（padding），读 Bs 时 BN 方向连续 → 消除 bank conflict
  // 相同参数：BM=BN=128, BK=8, TM=TN=8, 256 线程，WMITER=WNITER=1
  //
  // 【原因1：加载阶段边界检查的代价】
  //   kernel 8 (gemmResolveBankExtraCol_v2)：
  //     加载 As 时有多层降级边界检查（float4 → float3 → float2 → float1 → 零）：
  //       if (row < M && col+3 < K) { float4 load }
  //       else if (row < M && col+2 < K) { float3 load }
  //       else if (row < M && col+1 < K) { float2 load }
  //       else if (row < M && col < K) { float1 load }
  //       else { 写零 }
  //     测试尺寸（128~4096，均为 BM/BN/BK 整数倍）永远不触发边界分支，
  //     但分支的存在有三类代价：
  //       a. 编译器必须为每条分支生成条件跳转指令 → 每次 load 多 4~5 条比较/跳转
  //       b. 多条代码路径迫使编译器保留更多中间寄存器值 → 寄存器压力上升
  //       c. 编译器难以跨分支做指令重排和流水线合并 → 生成 PTX 质量下降
  //
  //   kernel 9 v2 (gemmAutotuned_v2)：
  //     加载阶段无任何边界检查，直接 float4 加载：
  //       float4 loadVecAs = reinterpret_cast<float4 *>(...)[0];
  //     代码路径唯一 → 编译器可充分展开、重排、流水线合并 → 生成指令数最少
  //
  // 【原因2：边界检查代码复杂度影响寄存器分配】
  //   kernel 8 的多层 if-else 让编译器在寄存器分配时必须同时保活多个分支的变量
  //   （float4/float3/float2 的临时寄存器），即使这些分支实际不执行。
  //   kernel 9 v2 只有一条 float4 路径，寄存器压力更低，
  //   编译器有更多空闲寄存器用于缓存循环变量，减少指令数和内存访问。
  //
  // 结论：布局相同时，代码复杂度（边界检查的有无）决定编译器能生成多优质的 PTX，
  //       进而影响实际执行性能。去掉 kernel 8 的边界检查后两者性能应接近一致。
  // ────────────────────────────────────────────────────────────────────────────

  // ── 4096 维度性能低于 2048 的原因分析 ────────────────────────────────────────
  //
  // 【背景：算术强度与 Roofline 模型】
  //   Roofline 模型将性能上界分为两段：
  //     算力段：FLOP/s ≤ 峰值算力（计算密集区，受 CUDA core 数量限制）
  //     带宽段：FLOP/s ≤ 峰值带宽 × 算术强度（带宽密集区，受内存带宽限制）
  //   交叉点（ridge point）= 峰值算力 / 峰值带宽，约 38 FLOP/byte（本机估算值）
  //
  //   算术强度 I = FLOP / Bytes_from_DRAM：
  //     每个 BM×BN 块：执行 2*BM*BN*K 次 FLOP，
  //     从 DRAM 加载 (BM+BN)*K*4 字节（理想复用，每 BK 列加载一次 block tile）
  //     → I ≈ 2*BM*BN / ((BM+BN)*4) = 2*128*128 / (256*4) = 32 FLOP/byte
  //   32 < 38（ridge point）→ kernel 9 处于带宽受限区，FLOP/s 上界由带宽决定
  //
  // 【原因：L2 缓存命中率随矩阵尺寸下降】
  //   本机 L2 容量约 6 MB（sm_86，参考 deviceQuery 输出）。
  //   三个矩阵（A + B + C）的总数据量：
  //     M=2048：2048*2048*4*3 = 50 MB  → 远超 L2，但时间局部性使部分 block tile
  //             在 L2 中驻留（A 的列 tile / B 的行 tile 被多个 block 复用）
  //     M=4096：4096*4096*4*3 = 192 MB → L2 完全被击穿（thrashing），
  //             每次 block tile 加载几乎全部来自 DRAM（L2 命中率接近 0%）
  //
  //   两种尺寸均处于带宽受限区，但 2048 时部分访问可命中 L2（延迟 ~200 cycle）
  //   而非每次都等 DRAM（延迟 ~600 cycle），等效带宽略高于 4096。
  //   → 4096 的实测 GFLOP/s 低于 2048，但差距不大（均受制于峰值内存带宽上限）
  //
  // 【为何差距不显著】
  //   带宽受限区的上界是峰值带宽，而非 L2 带宽；
  //   两种尺寸都已触及或接近 DRAM 带宽瓶颈，L2 命中率的差异只体现为小幅波动。
  //   若算术强度 > 38（如 BK=64），则进入计算密集区，
  //   L2 命中率的差异对 GFLOP/s 影响将更不明显。
  //
  // 结论：BK=8 导致算术强度（32 FLOP/byte）低于 ridge point（38），
  //       kernel 处于带宽受限区；4096 矩阵完全击穿 L2（6 MB），
  //       实际访问 DRAM 频率高于 2048，导致等效带宽略低，GFLOP/s 小幅下降。
  //
  // 【4096 性能下降说明未达 compute-bound】
  //   4096 时性能下降的原因是 L2 命中率下降（带宽受限），而非撞上了 FPU 的天花板。
  //   更具体地说：在算术强度（BK 约分，无法提升）和 FPU 利用效率均无任何改善的前提下，
  //   矩阵尺寸从 2048 增大到 4096 唯一改变的是 L2 命中率下降（数据量远超 6 MB L2），
  //   等效可用带宽降低，导致 GFLOPS 小幅下滑。
  //   kernel 10 在 4096 维度上仍能进一步提升 GFLOPS，证明 FPU 仍有空缺未被填满。
  //   若已达真正 compute-bound，kernel 10 的任何优化均无法提升性能。
  //   两个现象合在一起，确认整个优化序列此时尚未达到真正的 compute-bound。
  // ────────────────────────────────────────────────────────────────────────────

  // ── 增大 BK 的收益与 compute-bound 的关系 ─────────────────────────────────────
  //
  // Autotuner 最优参数 BK=16（kernel 8 使用 BK=8），性能提升来自摊薄同步开销，而非算术强度：
  //   算术强度 = 2×BM×BN×BK / ((BM+BN)×BK×4) = 2×BM×BN / ((BM+BN)×4)，BK 约分消去
  //   BK 对算术强度无影响，收益来自：K/BK 次外层循环减少 → __syncthreads() 调用次数减半
  //   每次迭代有效计算量翻倍，同步固定开销占比下降 → 执行时间缩短
  //
  // 这说明 kernel 未达到真正的 compute-bound：
  //   kernel 7：FMA 掩盖 SMEM bank conflict → SMEM 延迟不是瓶颈（局部结论）
  //   kernel 9：增大 BK 仍有收益 → 同步开销仍占一定比例 → 全局未达峰值 FLOPS
  //   两者不矛盾：前者只说明 SMEM 层不是瓶颈，后者揭示了另一层（同步）仍有优化空间
  //
  // 真正的 compute-bound：任何层次优化（内存布局、同步频率、参数调整）均无法继续提升，
  //   FPU 利用率接近 100%，需用 ncu 查看 sm_active_cycles 等计数器验证。
  // ────────────────────────────────────────────────────────────────────────────

template <const int BM,const int BN, const int BK, const int TM,const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS,1) gemmAutotuned_v2 (int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {

  // 该block负责计算的起始行和列
  const uint initRow {blockIdx.y * BM};
  const uint initCol {blockIdx.x * BN};

  // 该block的静态SMEM
  __shared__ float As[BM * BK];
  // padding
  // 一个bank是4字节，4个bank就是16字节，合并128位指令，要求地址16字节对齐
  const uint extraCol {4};
  __shared__ float Bs[BK * (BN+extraCol)];

  // 加载阶段
  // 该线程负责加载的行和列
  // 只要线程0进行静态断言
  // 可以在runner.cu中进行断言，不要在kernel中做静态或者动态断言
  static_assert(BK % 4 == 0 && "列组不是整数");
  static_assert(K9_NUM_THREADS % (BK / 4) == 0 && "不能覆盖完整列组");
  static_assert(BM * BK % (K9_NUM_THREADS * 4) == 0 && "不能覆盖完整行");
  const uint loadAsColGroup = threadIdx.x % (BK / 4) ;
  const uint loadAsRow = threadIdx.x / (BK / 4) ;
  // ── 加载步长（每次循环覆盖的行数）────────────────────────────────────────────
  // 每次循环：K9_NUM_THREADS 个线程各自加载 1 个 float4，
  //   As 每行有 BK/4 个 float4 → 每次覆盖 K9_NUM_THREADS / (BK/4) 行。
  //
  // 【错误写法及其报错】
  //   曾误写为 BM*BK/(K9_NUM_THREADS*4)，该值是"需要几次循环"（循环次数），不是步长。
  //   以 BM=128, BK=8, K9_NUM_THREADS=256 为例：
  //     循环次数 = 128*8/(256*4) = 1   ← 只需 1 次
  //     正确步长 = 256/(8/4)    = 128  ← 每次覆盖 128 行
  //   将循环次数(1)当步长使用 → 循环运行 128 次：
  //     RowIdx=0 : As[(loadAsRow+0)*BK + ...], loadAsRow ∈ [0,127] → 正常
  //     RowIdx=1 : As[(loadAsRow+1)*BK + ...], loadAsRow=127 → As[128*8] 越界写入
  //   → 报错：[CUDA ERROR] at file gemm.cu:163:
  //            an illegal memory access was encountered
  //   根因：shared memory 越界写入，CUDA 运行时在下一个同步点（cudaDeviceSynchronize）
  //         检测到非法访问并通过 cudaGetLastError() 返回 cudaErrorIllegalAddress。
  // ────────────────────────────────────────────────────────────────────────────
  const uint loadAsRowNumberPerCycle {K9_NUM_THREADS / (BK / 4)};
  static_assert(BN % 4 == 0 && "列组不是整数");
  static_assert(K9_NUM_THREADS % (BN / 4) == 0 && "不能覆盖完整列组");
  static_assert(BN * BK % (K9_NUM_THREADS * 4) == 0 && "不能覆盖完整行");
  const uint loadBsColGroup = threadIdx.x % (BN / 4);
  const uint loadBsRow = threadIdx.x / (BN / 4);
  // Bs 每行有 BN/4 个 float4 → 每次覆盖 K9_NUM_THREADS / (BN/4) 行，同上推导。
  const uint loadBsRowNumberPerCycle {K9_NUM_THREADS / (BN / 4)};


  // 计算阶段
  // ── 曾犯错误1：static_assert 放在 if (threadIdx.x == 0) 内 ─────────────────
  // 曾误写为：
  //   if (threadIdx.x == 0) {
  //       static_assert(BK % 4 == 0 && "列组不是整数");
  //       static_assert(K9_NUM_THREADS % (BK / 4) == 0 && "不能覆盖完整列组");
  //   }
  // static_assert 是编译期指令，nvcc 实例化模板时立即求值，与任何运行期 if 无关。
  // if (threadIdx.x == 0) 是运行期条件，对编译期的 static_assert 完全无效——
  // 无论 if 条件是否成立，static_assert 都会在编译时执行（也可能永远不执行，取决于
  // 编译器是否认为该分支可达）。正确做法是将 static_assert 直接放在函数体顶层。
  // ────────────────────────────────────────────────────────────────────────────

  // 该线程在block中的二维分布
  constexpr uint threadColNum {16};
  const uint threadColGroup {threadIdx.x % threadColNum};
  const uint threadRowGroup {threadIdx.x / threadColNum};
  // block中的行
  static_assert(K9_NUM_THREADS % threadColNum == 0 && "行不是整数");
  // block中的线程有多少行
  constexpr uint threadRowNum {K9_NUM_THREADS / threadColNum};
  // 一个block中所有线程能够负责的行数WM和列数WN
  constexpr uint WN {threadColNum * TN};
  constexpr uint WM {threadRowNum * TM};
  // 暗含static_assert(BM >= WM && "一个block能处理的行数要超过BM");
  // 不需要边界检查
  static_assert(BM % WM == 0 && "能覆盖完整的行");
  // static_assert(BN >= WN && "一个block能处理的列出要超过BN");
  // 不需要边界检查
  static_assert(BN % WN == 0 && "能覆盖完整的列");
  // 该线程缓存的As、Bs元素
  float regAsCache[TM];
  float regBsCache[TN];
  // 需要几次迭代能覆盖所有的行和列
  const uint WMITER {BM / WM};
  const uint WNITER {BN / WN};

  // ── 曾犯错误2：threadResults 大小只够一个 warp tile ──────────────────────────
  // 曾误写为：float threadResults[TM * TN] {0.0f};
  // 每个线程参与 WMITER × WNITER 个 warp tile 的计算，需要为每个 tile 保存独立结果。
  // TM*TN 只够存一个 warp tile 的结果；WMITER>1 或 WNITER>1 时，
  // 不同 warp tile 的结果写入相同下标，相互覆盖 → 输出错误。
  // 正确大小：WMITER * WNITER * TM * TN（所有 warp tile 的结果总量）。
  //
  // ── 曾犯错误3：累积下标不含 warp tile 索引 ────────────────────────────────────
  // 曾误写为：threadResults[rowIdx * TN + colIdx] += ...;
  // 该下标完全忽略 wrowIdx / wcolIdx，不同 warp tile（wrowIdx=0, WM, 2*WM...）
  // 的结果都落入 [0, TM*TN) 同一段，各 tile 结果混叠累加 → 输出错误。
  // 正确写法：将 wrowIdx/wcolIdx 除以步长 WM/WN 换算成紧凑序号，再乘以 TM/TN 得偏移：
  //   threadResults[(wrowIdx/WM * TM + rowIdx) * (WNITER*TN) + wcolIdx/WN * TN + colIdx]
  // wrowIdx/WM ∈ {0,1,...,WMITER-1}（序号），*TM 得紧凑行偏移；wcolIdx/WN 同理。
  // ────────────────────────────────────────────────────────────────────────────

  // 该线程计算结果缓存
  // 当BM=WM,BN=WN时，WMITER=WNITER=1，就是之前kernel的情形
  float threadResults[WMITER * TM * WNITER * TN] {0.0f};


  // 向量写回C
  static_assert(TN % 4 == 0 && "列组不是整数");


  // 沿k方向循环
  for (uint outerColidx {}; outerColidx < K; outerColidx+=BK) {
    // 加载阶段
    for (uint RowIdx {}; RowIdx < BM; RowIdx+=loadAsRowNumberPerCycle) {
      // 加载As
      float4 loadVecAs { reinterpret_cast<float4 *>(&A[(initRow + loadAsRow + RowIdx) * K + loadAsColGroup * 4 + outerColidx])[0] };
      // 行主序
      // As[(loadAsRow + RowIdx) * BK + loadAsColGroup * 4] = loadVecAs.x;
      // As[(loadAsRow + RowIdx) * BK + loadAsColGroup * 4 + 1] = loadVecAs.y;
      // As[(loadAsRow + RowIdx) * BK + loadAsColGroup * 4 + 2] = loadVecAs.z;
      // As[(loadAsRow + RowIdx) * BK + loadAsColGroup * 4 + 3] = loadVecAs.w;

      // 列主序
      As[(loadAsColGroup * 4) * BM + loadAsRow + RowIdx] = loadVecAs.x;
      As[(loadAsColGroup * 4 + 1) * BM + loadAsRow + RowIdx] = loadVecAs.y;
      As[(loadAsColGroup * 4 + 2) * BM + loadAsRow + RowIdx] = loadVecAs.z;
      As[(loadAsColGroup * 4 + 3) * BM + loadAsRow + RowIdx] = loadVecAs.w;

    }

    // 加载Bs
    for (uint RowIdx {}; RowIdx < BK; RowIdx+=loadBsRowNumberPerCycle) {
      float4 loadVecBs { reinterpret_cast<float4 *>(&B[(loadBsRow + RowIdx + outerColidx) * N + initCol + loadBsColGroup * 4])[0] };
      // 行主序
      // Bs[(loadBsRow + RowIdx) * BN + loadBsColGroup * 4] = loadVecBs.x;
      // Bs[(loadBsRow + RowIdx) * BN + loadBsColGroup * 4 + 1]  = loadVecBs.y;
      // Bs[(loadBsRow + RowIdx) * BN + loadBsColGroup * 4 + 2]  = loadVecBs.z;
      // Bs[(loadBsRow + RowIdx) * BN + loadBsColGroup * 4 + 3]  = loadVecBs.w;

      // 行主序+paading
      Bs[(loadBsRow + RowIdx) * (BN + extraCol) + loadBsColGroup * 4] = loadVecBs.x;
      Bs[(loadBsRow + RowIdx) * (BN + extraCol) + loadBsColGroup * 4 + 1]  = loadVecBs.y;
      Bs[(loadBsRow + RowIdx) * (BN + extraCol) + loadBsColGroup * 4 + 2]  = loadVecBs.z;
      Bs[(loadBsRow + RowIdx) * (BN + extraCol) + loadBsColGroup * 4 + 3]  = loadVecBs.w;

    }

    __syncthreads();

    // ── 计算阶段：两种等价写法 ──────────────────────────────────────────────
    //
    // threadResults 大小：WMITER * WNITER * TM * TN
    //   每个线程的寄存器数组，存储该线程在所有 warp tile 中负责的输出元素。
    //
    // 【写法 A：绝对偏移循环（当前实现）】
    //   外层循环变量 wrowIdx/wcolIdx 是块内绝对偏移，步长 WM/WN。
    //   wrowIdx ∈ {0, WM, 2*WM, ..., (WMITER-1)*WM}，步长 WM = 16*TM。
    //   不能直接用 wrowIdx 做 threadResults 下标：
    //     相邻 tile 间有 (WM-TM)=15*TM 个空洞，下标不连续，
    //     最大值 ≈ BM * WNITER*TN，远超数组大小，越界。
    //   必须先除以步长换算成序号，再乘以每 tile 的行/列数得紧凑偏移：
    //     wmIdx  = wrowIdx / WM        ∈ {0, 1, ..., WMITER-1}
    //     wnIdx  = wcolIdx / WN        ∈ {0, 1, ..., WNITER-1}
    //     行偏移 = wmIdx * TM + rowIdx ∈ [0, WMITER*TM)
    //     列偏移 = wnIdx * TN + colIdx ∈ [0, WNITER*TN)
    //     下标   = 行偏移 * (WNITER*TN) + 列偏移
    //            = (wmIdx*TM + rowIdx) * (WNITER*TN) + wnIdx*TN + colIdx
    //     最大值 = WMITER*WNITER*TM*TN - 1  ✓
    //
    // 【写法 B：序号循环（等价，更直观，与 gemmAutotuned 第一个 kernel 一致）】
    //   外层直接用序号 wmIdx/wnIdx，无需除法换算，下标天然紧凑：
    //
    //   for (uint wmIdx {}; wmIdx < WMITER; ++wmIdx) {           // tile 行序号
    //     for (uint wnIdx {}; wnIdx < WNITER; ++wnIdx) {         // tile 列序号
    //       uint wrowOff = wmIdx * WM;   // 转回绝对偏移，用于访问 As
    //       uint wcolOff = wnIdx * WN;   // 转回绝对偏移，用于访问 Bs
    //       for (uint innerColIdx {}; innerColIdx < BK; ++innerColIdx) {
    //         for (uint rowIdx {}; rowIdx < TM; ++rowIdx)
    //           regAsCache[rowIdx] = As[(wrowOff + threadRowGroup*TM + rowIdx)*BK + innerColIdx];
    //         for (uint colIdx {}; colIdx < TN; ++colIdx)
    //           regBsCache[colIdx] = Bs[innerColIdx*BN + wcolOff + threadColGroup*TN + colIdx];
    //         for (uint rowIdx {}; rowIdx < TM; ++rowIdx)
    //           for (uint colIdx {}; colIdx < TN; ++colIdx)
    //             // wmIdx/wnIdx 已是序号，直接 *TM/*TN 得紧凑偏移，无需除法
    //             threadResults[(wmIdx*TM + rowIdx)*(WNITER*TN) + wnIdx*TN + colIdx]
    //                 += regAsCache[rowIdx] * regBsCache[colIdx];
    //       }
    //     }
    //   }
    //
    // 写法 A 和写法 B 下标公式完全相同（wmIdx == wrowIdx/WM），
    // 区别仅在循环变量形式：A 用绝对偏移需要除法换算，B 用序号天然紧凑。
    // ────────────────────────────────────────────────────────────────────────

    // 写法 A 实现
    for (uint wrowIdx {}; wrowIdx < BM; wrowIdx+=WM) {
      for (uint wcolIdx {}; wcolIdx < BN; wcolIdx+=WN) {
        for (uint innerColIdx{}; innerColIdx < BK; ++innerColIdx) {
          for (uint rowIdx {};rowIdx < TM; ++rowIdx) {
            // 行主序
            // regAsCache[rowIdx] = As[(wrowIdx + threadRowGroup * TM + rowIdx) * BK + innerColIdx];

            // 列主序
            regAsCache[rowIdx] = As[innerColIdx * BM + wrowIdx + threadRowGroup * TM + rowIdx];
          }
          for (uint colIdx {}; colIdx < TN; ++colIdx) {
            // 行主序
            // regBsCache[colIdx] = Bs[innerColIdx * BN + wcolIdx + threadColGroup * TN + colIdx];

            // 行主序+padding
            regBsCache[colIdx] = Bs[innerColIdx * (BN + extraCol) + wcolIdx + threadColGroup * TN + colIdx];
          }
          for (uint rowIdx {}; rowIdx < TM; ++rowIdx) {
            for (uint colIdx {}; colIdx < TN; ++colIdx) {
              // wrowIdx/WM → wmIdx（序号），*TM 得紧凑行偏移；wcolIdx/WN 同理, *TN得紧凑行偏移
              // 每个线程保存自己的计算结果
              threadResults[(wrowIdx/WM * TM + rowIdx) * (WNITER * TN) + wcolIdx/WN * TN + colIdx] += regAsCache[rowIdx] * regBsCache[colIdx];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // 向量写回C
  for (uint wrowIdx {}; wrowIdx < BM; wrowIdx+=WM) {
    for (uint wcolIdx {}; wcolIdx < BN; wcolIdx+=WN) {
      for (uint rowIdx{}; rowIdx < TM; ++rowIdx) {
        for (uint colGroupIdx{}; colGroupIdx < TN / 4; ++colGroupIdx) {
          float4 writeBackVec { reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * TM + wrowIdx + rowIdx) * N + initCol + threadColGroup * TN + wcolIdx + colGroupIdx * 4])[0] };
          writeBackVec.x =  alpha * threadResults[((wrowIdx / WM) * TM + rowIdx) * (WNITER * TN) + wcolIdx/WN * TN + colGroupIdx * 4] + beta * writeBackVec.x;
          writeBackVec.y =  alpha * threadResults[((wrowIdx / WM) * TM + rowIdx) * (WNITER * TN) + wcolIdx/WN * TN + colGroupIdx * 4 + 1] + beta * writeBackVec.y;
          writeBackVec.z =  alpha * threadResults[((wrowIdx / WM) * TM + rowIdx) * (WNITER * TN) + wcolIdx/WN * TN + colGroupIdx * 4 + 2] + beta * writeBackVec.z;
          writeBackVec.w =  alpha * threadResults[((wrowIdx / WM) * TM + rowIdx) * (WNITER * TN) + wcolIdx/WN * TN + colGroupIdx * 4 + 3] + beta * writeBackVec.w;
          reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * TM + wrowIdx + rowIdx) * N + threadColGroup * TN + wcolIdx + colGroupIdx * 4 + initCol])[0] = writeBackVec;
        }
      }
    }
  }
}
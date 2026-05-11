#!/usr/bin/env bash

set -u
set -o pipefail

# 搜索空间：每个模板参数的候选值列表
# gemmWarptiling_v2 模板参数：<BM, BN, BK, WM, WN, TM, TN>，NUM_THREADS 定义在 .cuh 中
BK_VALUES=(8 16 32 64)       # SMEM tile 在 K 方向的宽度
BM_VALUES=(64 128 256)       # SMEM block tile 在 M 方向的大小
BN_VALUES=(64 128 256)       # SMEM block tile 在 N 方向的大小
TM_VALUES=(4 8 16 32)        # 每个线程在 warp tile 内负责的行数
TN_VALUES=(4 8 16 32)        # 每个线程在 warp tile 内负责的列数
WM_VALUES=(32 64 128 256)    # warp tile 在 M 方向的大小（独立于 BM，可 < BM）
WN_VALUES=(32 64 128 256)    # warp tile 在 N 方向的大小（独立于 BN，可 < BN）
NUM_THREADS_VALUES=(128 256) # 每个 block 的线程数

cd "$(dirname "$0")"
cd ".."

RUNNER="src/runner.cu"
KERNEL="src/kernels/10_kernel_warptiling.cuh"
OUTPUT="autotune/kernel_10_autotune_results.txt"

mkdir -p "$(dirname "$OUTPUT")"
echo "" > $OUTPUT

export DEVICE="0"

# 实际值：2 × 4 × 4 × 4 × 3 × 3 × 4 × 4 = 18432（过滤前）
TOTAL_CONFIGS="$(( ${#NUM_THREADS_VALUES[@]} * ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} * ${#WM_VALUES[@]} * ${#WN_VALUES[@]} ))"
CONFIG_NUM=0

for bk in "${BK_VALUES[@]}"; do
  for tm in "${TM_VALUES[@]}"; do
    for tn in "${TN_VALUES[@]}"; do
      for bm in "${BM_VALUES[@]}"; do
        for bn in "${BN_VALUES[@]}"; do
          for wm in "${WM_VALUES[@]}"; do
            for wn in "${WN_VALUES[@]}"; do
              for nt in "${NUM_THREADS_VALUES[@]}"; do
                echo ""
                CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

                config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn WM=$wm WN=$wn NT=$nt"

                # ── 前置条件检查（与 kernel 源码 static_assert 一一对应）───────────────
                # kernel 10 无 kernel 9 的 threadColNum=16 硬编码，
                # 因此不需要 BN%(16*TN)==0 / BM%(16*TM)==0 约束。

                # As：向量加载列组是整数：BK % 4 == 0
                if [[ $(( $bk % 4 )) -ne 0 ]]; then
                  echo "VECTORIZE: Skipping $config because BK % 4 = $(( $bk % 4 )) != 0"
                  continue
                fi

                # As：block 能覆盖完整列组：NT % (BK/4) == 0，等价于 (NT*4) % BK == 0
                if [[ $(( ($nt * 4) % $bk )) -ne 0 ]]; then
                  echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BK = $(( ($nt * 4) % $bk )) != 0"
                  continue
                fi

                # As：block 能在 n 次迭代中覆盖完整行：(BM*BK) % (NT*4) == 0
                if [[ $(( ($bm * $bk) % (4 * $nt) )) -ne 0 ]]; then
                  echo "VECTORIZE: Skipping $config because (BM * BK) % (4 * NUM_THREADS) = $(( ($bm * $bk) % (4 * $nt) )) != 0"
                  continue
                fi

                # Bs：向量加载列组是整数：BN % 4 == 0
                if [[ $(( $bn % 4 )) -ne 0 ]]; then
                  echo "VECTORIZE: Skipping $config because BN % 4 = $(( $bn % 4 )) != 0"
                  continue
                fi

                # Bs：block 能覆盖完整列组：NT % (BN/4) == 0，等价于 (NT*4) % BN == 0
                if [[ $(( ($nt * 4) % $bn )) -ne 0 ]]; then
                  echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BN = $(( ($nt * 4) % $bn )) != 0"
                  continue
                fi

                # Bs：block 能在 n 次迭代中覆盖完整行：(BN*BK) % (NT*4) == 0
                if [[ $(( ($bn * $bk) % (4 * $nt) )) -ne 0 ]]; then
                  echo "VECTORIZE: Skipping $config because (BN * BK) % (4 * NUM_THREADS) = $(( ($bn * $bk) % (4 * $nt) )) != 0"
                  continue
                fi

                # warp tile 量化：WM 整除 BM，WN 整除 BN
                if [[ $(( $bm % $wm )) -ne 0 ]]; then
                  echo "TILING: Skipping $config because BM % WM = $(( $bm % $wm )) != 0"
                  continue
                fi

                if [[ $(( $bn % $wn )) -ne 0 ]]; then
                  echo "TILING: Skipping $config because BN % WN = $(( $bn % $wn )) != 0"
                  continue
                fi

                # 线程 tile 量化：TN 整除 WN，TM 整除 WM
                if [[ $(( $wn % $tn )) -ne 0 ]]; then
                  echo "TILING: Skipping $config because WN % TN = $(( $wn % $tn )) != 0"
                  continue
                fi

                if [[ $(( $wm % $tm )) -ne 0 ]]; then
                  echo "TILING: Skipping $config because WM % TM = $(( $wm % $tm )) != 0"
                  continue
                fi

                # 线程列组完整覆盖：NT % (WN/TN) == 0
                if [[ $(( $nt % ($wn / $tn) )) -ne 0 ]]; then
                  echo "TILING: Skipping $config because NUM_THREADS % (WN/TN) = $(( $nt % ($wn / $tn) )) != 0"
                  continue
                fi

                # 线程行迭代完整覆盖：WM % (NT/(WN/TN)*TM) == 0
                # threadrowNumPerIter = NT/(WN/TN)*TM，保证 threadRowIterNum >= 1 且整除
                threadrowNumPerIter=$(( ($nt / ($wn / $tn)) * $tm ))
                if [[ $(( $wm % $threadrowNumPerIter )) -ne 0 ]]; then
                  echo "TILING: Skipping $config because WM % (NT/(WN/TN)*TM) = $(( $wm % $threadrowNumPerIter )) != 0"
                  continue
                fi

                # ── sed 注入参数 ─────────────────────────────────────────────
                # runner.cu 中 K10 参数用 constexpr uint；
                # NUM_THREADS 定义在 kernel .cuh 中，单独替换。
                sed -i "s/constexpr uint K10_BK = .*/constexpr uint K10_BK = $bk;/" $RUNNER
                sed -i "s/constexpr uint K10_TM = .*/constexpr uint K10_TM = $tm;/" $RUNNER
                sed -i "s/constexpr uint K10_TN = .*/constexpr uint K10_TN = $tn;/" $RUNNER
                sed -i "s/constexpr uint K10_BM = .*/constexpr uint K10_BM = $bm;/" $RUNNER
                sed -i "s/constexpr uint K10_BN = .*/constexpr uint K10_BN = $bn;/" $RUNNER
                sed -i "s/constexpr uint K10_WM = .*/constexpr uint K10_WM = $wm;/" $RUNNER
                sed -i "s/constexpr uint K10_WN = .*/constexpr uint K10_WN = $wn;/" $RUNNER
                sed -i "s/constexpr uint K10_NUM_THREADS = .*/constexpr uint K10_NUM_THREADS = $nt;/" $KERNEL

                # ── 重新编译 ─────────────────────────────────────────────────
                if ! cmake --build cmake-build-release --target gemm -- -j 18 2>&1 | tee -a $OUTPUT; then
                  echo "COMPILE FAILED: $config" | tee -a $OUTPUT
                  continue
                fi

                echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn WM=$wm WN=$wn NUM_THREADS=$nt" |& tee -a $OUTPUT
                timeout -v 10 ./cmake-build-release/gemm 10 2>&1 | tee -a $OUTPUT
                echo "-------------------" | tee -a $OUTPUT
                echo "" | tee -a $OUTPUT
              done
            done
          done
        done
      done
    done
  done
done

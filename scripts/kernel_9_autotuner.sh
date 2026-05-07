#!/usr/bin/env bash

# 遇到未定义的变量时立即报错退出，防止变量名拼写错误导致静默错误
set -u
# 管道退出码取第一个失败命令的退出码（默认是最后一条命令的退出码）：
#   不加 pipefail：pipeline = "cmake ... | tee"，退出码 = tee 的退出码（几乎恒为 0）
#                  if ! pipeline → ! 0 = 真 → 无论 cmake 成功失败都进 then 块，条件判断失效
#   加 pipefail  ：cmake 失败 → pipeline 退出码 = cmake 的非零退出码
#                  if ! pipeline → ! 非零 = 真 → 正确进入 then 块
set -o pipefail

# 搜索空间：每个模板参数的候选值列表
# 自动调参对所有参数组合做全量网格搜索（暴力枚举）
BK_VALUES=(8 16 32 64)       # SMEM tile 在 K 方向的宽度（As 和 Bs 共用）
TM_VALUES=(4 8 16 32)        # 每个线程在输出 tile 中负责计算的行数
TN_VALUES=(4 8 16 32)        # 每个线程在输出 tile 中负责计算的列数
BM_VALUES=(64 128 256)       # SMEM block tile 在 M 方向的大小
BN_VALUES=(64 128 256)       # SMEM block tile 在 N 方向的大小
NUM_THREADS_VALUES=(256)     # 每个 block 的线程数（只有一个候选值，固定不搜索）

# 切换到项目根目录，使后续所有相对路径以项目根为基准
# $0        : 脚本自身的路径（由调用方式决定，可能是绝对路径或相对路径）
#             例：./scripts/kernel_9_autotuner.sh 或 /home/.../scripts/kernel_9_autotuner.sh
# dirname   : 取路径的目录部分，去掉最后一个 / 及其后的文件名
#             dirname "$0" → .../GEMM_CUDA/scripts/
# $(...)    : 命令替换，将 dirname 的输出作为字符串传给 cd
# 第一条 cd : 切到脚本所在目录（scripts/），而非调用时的当前目录
#             无论从哪个目录调用本脚本，路径都能正确解析
# 第二条 cd : 退一级，到达项目根目录（GEMM_CUDA/）
#             两条 cd 合在一起才完成切换，$0 本身不指向项目根
cd "$(dirname "$0")"
cd ".."

# 需要被 sed 原地修改的源文件路径
RUNNER="src/runner.cu"
KERNEL="src/kernels/9_kernel_autotuned.cuh"
OUTPUT="benchmark_results/kernel_9_autotune_results.txt"

# 清空结果文件，避免将本次结果追加到上一次残留数据之后
# >  覆盖写（截断）：打开文件时先将文件截断为 0 字节，再写入，原有内容全部丢失
# >> 追加写        ：保留原有内容，写入指针定位到文件末尾，新内容追加在后
# |  管道          ：目标不是文件，而是另一个进程的 stdin；
#                   数据存在内核缓冲区（内存），进程结束即消失，从不落盘；
#                   无覆盖/追加的概念，> / >> 是 stdout→文件，| 是 stdout→stdin
echo "" > $OUTPUT

# 指定使用哪块 GPU，通过环境变量而非 argv 传递：
#   export 将变量提升为环境变量，shell fork 子进程时自动继承父进程的环境变量表，
#   gemm 内部用 getenv("DEVICE") 读取，不经过 main() 的 argc/argv。
#   不加 export 只是 shell 局部变量，子进程不可见。
export DEVICE="0"

# 计算参数组合总数，用于进度提示
# 语法解析：
#   "..."           : 外层引号防止结果被单词分割（数字不会被分割，属防御性写法）
#   $(( ))          : 算术展开，计算括号内整数表达式并返回结果；
#                     区别于 $( )（命令替换，运行命令取 stdout）
#   ${#ARRAY[@]}    : 取数组元素个数
#                     [@] —— 引用数组全部元素
#                     # 前缀对数组使用时 = 元素个数；对普通字符串使用时 = 字符数
#   *               : 算术展开内的乘法
# 实际值：1 × 4 × 4 × 4 × 3 × 3 = 576
TOTAL_CONFIGS="$(( ${#NUM_THREADS_VALUES[@]} * ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} ))"
CONFIG_NUM=0

# 枚举所有参数组合（六层嵌套循环）
for bk in ${BK_VALUES[@]}; do
  for tm in ${TM_VALUES[@]}; do
    for tn in ${TN_VALUES[@]}; do
      for bm in ${BM_VALUES[@]}; do
        for bn in ${BN_VALUES[@]}; do
          for nt in ${NUM_THREADS_VALUES[@]}; do
            echo ""
            CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

            config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NT=$nt"

            # ── 前置条件检查 ────────────────────────────────────────────────
            # 所有参数都是编译期常量，非法组合会导致错误结果或编译失败，
            # 因此在编译前用数学条件提前过滤掉不合法的配置。
            #
            # ⚠ 已知缺失约束（导致结果文件中出现大量 Divergence!）：
            #   本脚本只检查了向量加载对齐和 warp tile 量化这 6 个条件，
            #   但漏掉了"矩阵尺寸必须是 tile 的整数倍"这一前提：
            #     matrix_size % BM == 0
            #     matrix_size % BN == 0
            #
            #   Kernel 9 内部无任何边界守卫（无 if (globalRow < M) 之类的检查），
            #   它的正确性完全依赖调用方保证 M%BM==0 且 N%BN==0。
            #   违反此条件时发生两类错误：
            #
            #   【直接越界】BN=256 测试 size=128（BN > N）：
            #     加载 Bs 时读 B 的第 128..255 列（越界，取得垃圾值）；
            #     写 C 时 wnIdx=2,3 的列偏移也越界，覆盖相邻内存。
            #     → size=128 立即 Divergence（误差数十～数百）
            #
            #   【跨 size 污染】BM=256 测试 size=128（BM > M）：
            #     runner.cu 分配 max_size×max_size 的大缓冲区，所有 size 共用。
            #     size=128 时 wmIdx=2,3 越界写入 flat index 16384..（128-wide layout 的 C[128..][...]）；
            #     size=256 时同一 flat index 16384 被解读为 C[64][0]（256-wide layout），
            #     kernel 读取 beta*C 项时得到被污染的巨大值。
            #     → size=128 通过验证（verify 只检查 0..16383），size=256 出现天文数字偏差

            # GMEM→SMEM 加载 As tile（BM×BK）：
            #   每行有 BK/4 个 float4 位置；NT 线程各取一个 float4，
            #   每次迭代覆盖 NT/(BK/4) = (NT*4)/BK 行（即 rowStrideA）。
            #   rowStrideA 必须是整数，否则某些线程跨行加载，导致越界或数据错误。
            #   约束：(NT*4) % BK == 0，即 NT 是 BK/4 的整数倍。
            #
            # bash 语法解析：
            #   [[ ]]     : bash 扩展条件表达式，比 [ ] 更安全（支持 &&/|| 且字符串比较无需引号）
            #   $(( ))    : 算术展开，括号内 % 是取模，变量可不加 $（$nt 和 nt 均合法）
            #   -ne       : numeric not equal，数值不等于（区别于字符串比较 !=）
            if [[ $(( ($nt * 4) % bk )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BK = $(( ($nt * 4) % bk )) != 0))"
              continue
            fi

            # GMEM→SMEM 加载 Bs tile（BK×BN）：
            #   Bs 每行有 BN/4 个 float4 列组；NT 线程各取一个 float4，
            #   每次迭代覆盖 NT/(BN/4) = (NT*4)/BN 行（即 rowStrideB）。
            #   NT 必须是 BN/4 的整数倍，保证每次迭代恰好覆盖完整的列组，无跨行或余数。
            #   约束：(NT*4) % BN == 0（与 As 约束同理，除数从 BK 改为 BN）。
            if [[ $(( ($nt * 4) % bn )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BN = $(( ($nt * 4) % bn )) != 0))"
              continue
            fi


            # As tile 行间覆盖：tile 共 BM*BK 个元素，每次迭代 NT 线程各取一个 float4，
            #   消耗 NT*4 个元素，总迭代次数 = BM*BK/(NT*4) 必须是整数，
            #   等价于 BM % rowStrideA == 0：循环结束后 BM 行全部覆盖，无遗漏行。
            #   与前两个约束的分工：
            #     (NT*4) % BK == 0     → 每次迭代内 NT 线程覆盖整数行（行内无碎片）
            #     (BM*BK) % (NT*4) == 0 → 循环跑完后 BM 行全部覆盖（行间无遗漏）
            if [[ $(( ($bm * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BM * BK) % (4 * NUM_THREADS) = $(( ($bm * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi

            # Bs tile 行间覆盖：同上，约束改为 BN*BK % (NT*4) == 0，
            #   保证循环结束后 BK 行（每行 BN/4 个 float4）全部覆盖，无遗漏行。
            if [[ $(( ($bn * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BN * BK) % (4 * NUM_THREADS) = $(( ($bn * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi

            # Warp tile 在 N 方向的量化（完整列覆盖）：
            #   两种设计思路，目标相同（保证计算时完整覆盖列方向），但出发点相反：
            #
            #   【Kernel 9 的做法：先固定 blockDim=256=16×16，倒推 BN 约束】
            #     WN = TN*16 硬编码，threadCol = threadIdx % 16 固定为 [0,16)，与 BN 无关。
            #     Bs 访问最大 N 偏移 = 15*TN+(TN-1) = WN-1。
            #     要求 WNITER*WN == BN，即 BN % (TN*16) == 0，否则最后一次 wnIdx 迭代越界。
            #
            #   【等价替代：先有 BN，再推 blockDim 约束】
            #     BN % TN == 0                     → 列组是整数
            #     blockDim % (BN/TN) == 0          → 线程能覆盖完整列组
            #     两套条件目标相同但并不等价：
            #     反例 BN=64,TN=8：替代条件均满足，但 threadCol 最大访问第127列 > BN-1=63，越界。
            #     Kernel 9 更严格，因为 threadCol 范围由 WN 固定，与 BN 无关。
            if [[ $(( $bn % (16 * $tn ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BN % (16 * TN) = $(( $bn % (16 * $tn ) )) != 0))"
              continue
            fi

            # Warp tile 在 M 方向的量化（完整行覆盖）：同上，WM = TM*16 硬编码，
            #   threadRow = threadIdx/16 固定为 [0,16)，BM % (TM*16) == 0，
            #   替代条件为 BM%TM==0 且 blockDim%(BM/TM)==0，同样不等价，理由与 N 方向相同。
            if [[ $(( $bm % (16 * $tm ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BM % (16 * TM) = $(( $bm % (16 * $tm ) )) != 0))"
              continue
            fi

            # ── 用 sed 将参数原地写入源文件 ────────────────────────────────
            # 参数不通过 argv 传给可执行文件，而是在编译前直接注入源码。
            # runner.cu 中以 `const uint K9_XX = <旧值>;` 的形式定义这些常量，
            # sed -i 用正则 `.*` 匹配行尾的任意内容（含旧值和分号）并整行替换。
            # bash 在调用 sed 之前已将 $bk/$tm/... 展开为具体数字，
            # 因此 sed 看到的是字面字符串，例如：
            #   s/const uint K9_BK = .*/const uint K9_BK = 16;/
            # 注意：runner.cu 中有同名注释行也会被匹配并替换，
            # 但替换后仍是注释，对编译无影响。
            sed -i "s/const uint K9_BK = .*/const uint K9_BK = $bk;/" $RUNNER
            sed -i "s/const uint K9_TM = .*/const uint K9_TM = $tm;/" $RUNNER
            sed -i "s/const uint K9_TN = .*/const uint K9_TN = $tn;/" $RUNNER
            sed -i "s/const uint K9_BM = .*/const uint K9_BM = $bm;/" $RUNNER
            sed -i "s/const uint K9_BN = .*/const uint K9_BN = $bn;/" $RUNNER
            # NUM_THREADS 定义在 kernel 头文件中而非 runner.cu，单独替换
            sed -i "s/const int K9_NUM_THREADS = .*/const int K9_NUM_THREADS = $nt;/" $KERNEL

            # ── 重新编译 ────────────────────────────────────────────────────
            # sed 写入磁盘后同步执行编译：nvcc 读取已修改的文件，
            # 将上述常量作为模板实参编译进二进制，值变为汇编指令里的立即数。
            # 模板参数必须是编译期常量，每个配置都需要完整重编译，无法运行时切换。
            # 注意脚本未设 set -e，编译失败不会中断循环，会继续执行 ./gemm（通常崩溃）。
            # cmake --build 是对底层构建工具的封装，本项目生成器为 Unix Makefiles，
            # 实际执行：gmake -C cmake-build-release gemm -j 18
            # 相比直接 make，cmake --build 无需 cd 进构建目录，路径作参数传入即可。
            #
            # 不加 set -e 而是显式检查退出码：
            #   set -e 会在第一次编译失败时中断整个脚本，导致后续所有配置都不运行。
            #   编译失败（如 SMEM 超出硬件上限）在网格搜索中是预期内的事，应跳过该配置继续搜索。
            #   若编译失败后不 continue 直接运行 ./gemm，跑的是上一轮的旧二进制，结果文件会被污染。
            # 2>&1 | tee -a $OUTPUT：
            #   cmake 的 stdout 和 stderr 均进管道，tee 同时写到终端和 $OUTPUT（追加）。
            # if ! pipeline：
            #   有 pipefail：退出码 = cmake 的退出码；cmake 失败 → 进 then 块 ✓
            #   无 pipefail：退出码 = tee 的退出码（恒 0）  → if ! 0 = 真 → 始终进 then 块 ✗
            if ! cmake --build cmake-build-release --target gemm -- -j 18; then
            # if ! cmake --build cmake-build-release --target gemm -- -j 18 2>&1 | tee -a $OUTPUT; then
              # cmake 与 echo 是两条独立命令，先后执行，不共享管道：
              #   cmake 的 stderr 已在上一行随管道写入终端和 $OUTPUT，此时 cmake 已结束。
              #   此处 echo 只是追加一条人读的失败摘要，| 与 |& 等价（echo 无 stderr）。
              echo "COMPILE FAILED: $config" | tee -a $OUTPUT
              continue
            fi

            # echo 将当前进度和参数组合打印到 stdout，bash 在执行前展开所有 $变量。
            # |& 是 bash 对 2>&1 | 的缩写，含义：
            #   |      : 建立管道，将左侧命令的 stdout（fd 1）接入管道
            #   2>&1   : 将 stderr（fd 2）重定向到 fd 1 当前指向的位置（即管道入口）
            #            顺序关键：必须先 | 建立管道，再 2>&1 跟上，否则 fd 1 还指向终端
            #            shell fork 子进程前先在 OS 层建立管道，子进程内 fd 1 已指向管道写端，
            #            2>&1 再将 fd 2（stderr）重定向到 fd 1 当前所指（即管道写端），
            #   结果   : stdout 和 stderr 都进入管道，由右侧命令（tee）统一读取
            #   管道始终由 shell 优先建立，与 2>&1 在文本中的位置无关。
            # echo 不产生 stderr，此处 |& 与 | 效果相同，写法出于防御性习惯。
            # tee -a $OUTPUT：将 stdin 同时写到两个地方：
            #   终端（stdout）—— 实时看到进度
            #   $OUTPUT 文件  —— -a 追加模式，不覆盖已有内容
            echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NUM_THREADS=$nt" |&  tee -a $OUTPUT
            # timeout -v 10 ./cmake-build-release/gemm 9 | tee -a $OUTPUT
            #   timeout    : 以时间限制运行子进程，超时后发送 SIGTERM 将其杀死
            #   -v         : verbose，超时被杀时向 stderr 打印提示信息：
            #                  "Timeout: Sending signal TERM to command './cmake-build-release/gemm'"
            #                stderr 未被 | 捕获，直接打印到终端，不写入 $OUTPUT
            #   10          : 时间限制 10 秒，监控 gemm 整个进程的挂钟时间（wall-clock）：
            #               gemm 9 的执行流程为：分配GPU内存 → warm-up → 计时循环 → 打印结果
            #               极慢配置（寄存器溢出/SMEM超限）每次 kernel 调用耗时极长，
            #               计时循环无法在 10 秒内完成，整个进程被 SIGTERM 杀死
            #   ./cmake-build-release/gemm : "./" 是执行时的当前工作目录（脚本开头已 cd 到项目根），
            #                               完整路径为 <项目根>/cmake-build-release/gemm
            #                               由 add_executable(gemm ...) 生成
            #   9          : argv[1]，传给 main()，选择运行第 9 号 kernel
            #   |          : 只管道 stdout（gemm 的 benchmark 输出），stderr（timeout 的 -v 信息）不进入管道
            #   tee -a $OUTPUT : tee 将管道读到的内容同时写到两处（T型三通）：
            #                     └─ 自己的 stdout → 终端（实时可见）
            #                     └─ $OUTPUT 文件  → 追加写入
            #                   进入管道（即写入终端和$OUTPUT）的内容仅来自 gemm 的 stdout：
            #                     printf("Running kernel %d...\n")       gemm.cu:63  每次启动打印一次
            #                     std::cout << "dimensions(m=n=k) ..."   gemm.cu:147 每个矩阵尺寸打印一次
            #                     std::cout << "Failed to pass..."        gemm.cu:180 验证失败时打印
            #                     printf("Average elapsed time: ...")     gemm.cu:265 每个矩阵尺寸打印一次
            #                   gemm 内部另有 fs（std::ofstream）写 kernel_9_result.txt，
            #                   fs 与 stdout 是两条独立的输出流，完全绕过管道和 tee，
            #                   不进终端，也不进 $OUTPUT。
            #                   注意：终端上的 "Running kernel 9 on device 0." 来自 printf（stdout），
            #                   不是 fs << "Running kernel..."（两处写了内容相同的字符串，目标不同）。
            # 超过10秒的就不计入统计
            timeout -v 10 ./cmake-build-release/gemm 9 | tee -a $OUTPUT
            echo "-------------------" | tee -a $OUTPUT
            echo "" | tee -a $OUTPUT
          done
        done
      done
    done
  done
done

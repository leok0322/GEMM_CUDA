#!/bin/bash

# run_all_kernels.sh：对 kernel 0~12 依次执行，输出结果保存到 benchmark_results/
#
# 用法：
#   chmod +x scripts/run_all_kernels.sh
#   ./scripts/run_all_kernels.sh
#
# 输出：benchmark_results/kernel<i>_result.txt，每个 kernel 一个文件

# $0：当前脚本的路径；dirname 取其所在目录；realpath 转为绝对路径
# 以脚本所在目录的上一级（项目根）为基准，无论从哪里执行脚本路径都正确
PROJECT_ROOT="$(realpath "$(dirname "$0")/..")"
BINARY="$PROJECT_ROOT/cmake-build-release/gemm"
OUTPUT_DIR="$PROJECT_ROOT/benchmark_results"
ERROR_LOG_DIR="$PROJECT_ROOT/logs"

# mkdir -p：递归创建目录，已存在则不报错（幂等）
mkdir -p "$OUTPUT_DIR"
mkdir -p "$ERROR_LOG_DIR"

# seq 0 12：生成 0 1 2 ... 12 的序列，for 循环依次赋值给 i
for i in $(seq 9 9); do
    echo "Running kernel $i ..."
    # DEVICE=0        ：shell 内联环境变量语法，仅对紧跟的这条命令生效，不影响后续命令
    #                   等价于 export DEVICE=0 + 执行 + unset DEVICE，但更简洁
    # "$BINARY" "$i"  ：加引号防止路径含空格时被 shell 拆分为多个参数
    #                   ${i} 花括号明确变量名边界，避免歧义
    # 2>&1            ：将 stderr 重定向到 stdout 当前指向的位置（此处为终端）
    #                   顺序重要：必须先 >file 再 2>&1 才能把两者都写入文件
    #                   原因：2>&1 是时间点快照，stderr 指向 stdout 当时的目标，后续 stdout 变化不会跟着改变
    #                     正确：> file → stdout=file；  2>&1 → stderr=file  ✓
    #                     错误：2>&1  → stderr=终端；  > file → stdout=file  ✗
    # chmod +x        ：给脚本添加可执行权限，等价于 chmod 755
    #                   Linux 新建文件默认无可执行权限（-rw-r--r--），加 +x 后变为 -rwxr-xr-x
    #                   r=4, w=2, x=1，755 = rwx(7) r-x(5) r-x(5)
    # 2> "$file" ：单独重定向 stderr 到文件，stdout 仍输出到终端，两路分开
    #              stderr 包含 cudaCheck 错误、verify_matrix 失败信息等
    # 对比：
    #   2> "$file"  → stderr 重定向到文件（fd 2 → 磁盘文件）
    #   2>&1        → stderr 重定向到 stdout 当前指向的位置（fd 2 → fd 1，目标随 fd 1 走）
    #   两者的 "2>" 含义相同（重定向 fd 2），区别只在目标：
    #     "$file" 是具体路径（磁盘文件）
    #     &1      是 fd 1 的当前目标（可能是终端、管道或文件，取决于 fd 1 的状态）
    #
    # ── 当前命令解析 ─────────────────────────────────────────────────────────
    # DEVICE=0 "$BINARY" "$i" 2>&1 | tee -a "$ERROR_LOG_DIR/kernel${i}_error.txt"
    #
    # 建立顺序（bash 在 fork 前完成）：
    #   1. | 先在 OS 层建立管道，fd 1 → pipe 写端
    #   2. 2>&1：将 fd 2 重定向到 fd 1 当前所指（pipe 写端）
    #      → stdout 和 stderr 合并进同一条管道
    #   3. tee 从管道读取，同时写到终端（自身 stdout）和文件（追加）
    #
    # 数据流：
    #   binary stdout ──┐
    #                   ├──→ pipe ──→ tee ──┬──→ 终端
    #   binary stderr ──┘  (合并)           └──→ error.txt（-a 追加）
    #
    # stdout 和 stderr 交织在同一文件中，无法单独提取，但实现最简单。
    #
    # ── 若要 stdout 和 stderr 分离写入不同目标 ──────────────────────────────
    # 需用进程替换（process substitution），将 stderr 单独接入另一个 tee：
    #
    #   DEVICE=0 "$BINARY" "$i" \
    #       2> >(tee "$ERROR_LOG_DIR/kernel${i}_error.txt" >&2) \
    #       | tee -a "$OUTPUT_DIR/kernel${i}_result.txt"
    #
    #   | 只连接左侧的 fd 1（stdout），stderr（fd 2）完全不受影响：
    #     cmd1 的 fd 1 → 管道写端 → cmd2 的 fd 0
    #     cmd1 的 fd 2 → 不动，仍指向原来的目标（默认终端）
    #
    #   2> >(tee -a "error.txt" >&2)：
    #     >(...)  : bash 进程替换，bash 创建一个子进程，返回其 stdin 对应的 fd 路径
    #               （如 /dev/fd/63），子进程等待从该 fd 读数据
    #     2>      : 将 binary 的 stderr（fd 2）重定向到上面那个 fd 路径
    #               即：binary 的 stderr 成为进程替换子进程的 stdin
    #     tee     : 从 stdin 读取（即 binary 的 stderr），同时写到文件和自身 stdout
    #     -a      : 追加写，保留历次运行的错误记录
    #     >&2     : 将 tee 的 stdout（fd 1）重定向到 fd 2（终端）
    #               完整写法是 1>&2，> 左侧不写数字时默认为 fd 1，故 >&2 == 1>&2
    #               必要性：| 建立管道后，进程替换子进程继承了 fd 1 → 管道写端；
    #               若不加 >&2，tee 的 stdout 会进入主管道，与 binary 的 stdout 混入
    #               同一个 result.txt，污染结果文件；
    #               加了 >&2 后，tee 的 stdout "从管道里逃出去"，借道 fd 2 回到终端。
    #
    #   数据流：
    #     binary stdout ──────────────────────→ pipe → tee -a → 终端 + result.txt
    #     binary stderr → 进程替换子进程 stdin
    #                         ↓ tee 读取
    #                     ├──→ error.txt（-a 追加）
    #                     └──→ >&2 → 终端（不进主管道）
    #
    # ── 三种写法对比 ──────────────────────────────────────────────────────────
    #   2> file                         : stdout→终端，stderr→文件（只）
    #   2>&1 | tee -a file              : stdout+stderr→终端+文件（合并）
    #   2>>(tee file >&2) | tee -a res  : stdout→终端+result.txt，stderr→终端+error.txt（分离）

    # DEVICE=0 "$BINARY" "$i" 2> "$ERROR_LOG_DIR/kernel${i}_error.txt"
    # DEVICE=0 "$BINARY" "$i"  2> >(tee  -a "$ERROR_LOG_DIR/kernel${i}_error.txt" >&2) | tee -a "$OUTPUT_DIR/kernel${i}_result.txt"
    DEVICE=0 "$BINARY" "$i" 2>&1 | tee -a "$ERROR_LOG_DIR/kernel_${i}_error.txt"
    echo "  -> errors: $ERROR_LOG_DIR/kernel_${i}_error.txt"
done

echo "Done. Results in $OUTPUT_DIR/"

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
ERROR_LOG_DIR="$PROJECT_ROOT/error_logs"

# mkdir -p：递归创建目录，已存在则不报错（幂等）
mkdir -p "$OUTPUT_DIR"
mkdir -p "$ERROR_LOG_DIR"

# seq 0 12：生成 0 1 2 ... 12 的序列，for 循环依次赋值给 i
for i in $(seq 7 7); do
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
    # 2> ：单独重定向 stderr 到 error_logs/，与 stdout（终端）分开
    #      stderr 包含 cudaCheck 错误、verify_matrix 失败信息等
    DEVICE=0 "$BINARY" "$i" 2> "$ERROR_LOG_DIR/kernel${i}_error.txt"
    echo "  -> errors: $ERROR_LOG_DIR/kernel${i}_error.txt"
done

echo "Done. Results in $OUTPUT_DIR/"

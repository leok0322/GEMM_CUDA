#!/bin/bash

# run.sh：项目根目录入口脚本，依次执行所有 kernel 并进行 ncu 性能采集
#
# 用法：
#   chmod +x run.sh
#   ./run.sh

SCRIPT_DIR="$(realpath "$(dirname "$0")/scripts")"

echo "=== Running all kernels ==="
bash "$SCRIPT_DIR/run_all_kernels.sh"

#echo ""
#echo "=== Profiling all kernels with ncu ==="
#bash "$SCRIPT_DIR/ncu_profile_all.sh"  # ncu 免密已在脚本内部通过 sudoers 处理

echo ""
echo "All done."

#!/bin/bash

# ncu_profile_all.sh：对 kernel 0~12 依次执行 ncu 性能采集，结果保存到 benchmark_results/
#
# 用法：
#   chmod +x scripts/ncu_profile_all.sh      （添加可执行权限，等价于 chmod 755）
#                                              r=4,w=2,x=1；755=rwx(7)r-x(5)r-x(5)
#                                              Linux 新建文件默认无 x 权限，必须手动添加
#   sudo ./scripts/ncu_profile_all.sh        （ncu 采集硬件计数器通常需要 root 权限）
#
# 输出：benchmark_results/ncu_kernel<i>_result.txt，每个 kernel 一个文件


# 注意：
#   - 需使用 Release 构建（cmake-build-release），Debug 构建含 -G 禁止优化，性能数据无参考价值
#   - --set full 会采集所有指标，每个 kernel 耗时较长
#   - ncu 采集硬件计数器需要 root 权限，否则报 ERR_NVGPUCTRPERM




# $0：当前脚本的路径；dirname 取其所在目录；realpath 转为绝对路径
# 以脚本所在目录的上一级（项目根）为基准，无论从哪里执行脚本路径都正确
PROJECT_ROOT="$(realpath "$(dirname "$0")/..")"
BINARY="$PROJECT_ROOT/cmake-build-release/gemm"
OUTPUT_DIR="$PROJECT_ROOT/benchmark_results"

# mkdir -p：递归创建目录，已存在则不报错（幂等）
mkdir -p "$OUTPUT_DIR"

# seq 0 12：生成 0 1 2 ... 12 的序列，for 循环依次赋值给 i
for i in $(seq 1 2); do
    echo "Profiling kernel $i ..."
    # ncu --set full               ：采集所有预定义指标集（内存带宽、计算吞吐、warp效率等）
    # --print-summary per-kernel   ：按 kernel 为单位打印汇总统计
    # "$BINARY" "$i"               ：ncu 启动目标程序并透传参数；加引号防止路径含空格时出错
    #                                ${i} 花括号明确变量名边界，避免 $i_result 被误解为变量名
    # > "$OUTPUT_DIR/..."          ：stdout 重定向，覆盖写入文件
    # 2>&1                         ：将 stderr 重定向到 stdout 当前指向的文件
    #                                效果：stdout 和 stderr 都写入同一文件，ncu 警告/错误不丢失
    #                                顺序重要：必须先 >file 再 2>&1，反之 stderr 仍输出到终端
    #                                原因：2>&1 是时间点快照，把 stderr 指向 stdout 当时的目标
    #                                  正确：> file    → stdout=file；  2>&1 → stderr=file   ✓
    #                                  错误：2>&1      → stderr=终端；  > file → stdout=file  ✗
    #                                  stderr 不会随 stdout 后续变化而跟着改变

    # sudo /usr/local/cuda/bin/ncu：以 root 权限运行 ncu
    #   原因：ncu 采集 GPU 硬件性能计数器（Hardware Performance Counters）时
    #         需要直接访问 /dev/nvidia* 设备和内核 perf 子系统，普通用户无此权限
    #         不加 sudo 会报：ERR_NVGPUCTRPERM - The user does not have permission
    #                         to access NVIDIA GPU Performance Counters
    #
    #   必须用绝对路径，不能写 sudo ncu：
    #     sudo 使用自己的受限 PATH，不包含 /usr/local/cuda/bin
    #     写 sudo ncu 会报：sudo: ncu: command not found
    #     写绝对路径则直接定位可执行文件，不依赖 PATH
    #     sudoers 白名单也要求绝对路径，两者保持一致
    #
    #   免密配置（一次性，手动执行）：
    #     echo 'liam ALL=(ALL) NOPASSWD: /usr/local/cuda/bin/ncu' | sudo tee /etc/sudoers.d/ncu
    #     sudo chmod 440 /etc/sudoers.d/ncu
    #   含义：liam 执行 /usr/local/cuda/bin/ncu 时免密，其他 sudo 命令不受影响
    #   配置后 sudo /usr/local/cuda/bin/ncu 不再弹出密码提示


    # 【WSL2 限制：ncu 所有硬件计数器功能均不可用】
    # ncu 采集性能计数器需要直接访问 GPU 硬件寄存器，WSL2 内核不支持此功能
    # 无论 --set full 还是 --metrics 单个指标，均报 ERR_NVGPUCTRPERM
    # 与权限无关，加 sudo 也无效，是 WSL2 内核层的硬性限制，无法绕过
    #
    # 解决方案：
    #   方案一：在 Windows 下用 Nsight Compute GUI 远程连接 WSL2 里的程序分析
    #   方案二：装双系统，在原生 Linux 下运行 ncu
    #   方案三：直接用程序自身输出的 GFLOPS 数据，不依赖 ncu
    #
    # 以下命令在原生 Linux 下可用，WSL2 下会报错：
    sudo DEVICE=0 /usr/local/cuda/bin/ncu --set full --print-summary per-kernel "$BINARY" "$i" > "$OUTPUT_DIR/ncu_kernel${i}_result.txt" 2>&1
    echo "  -> saved to $OUTPUT_DIR/ncu_kernel${i}_result.txt"
done

echo "Done. Results in $OUTPUT_DIR/"



#      liam@leo:~/cpp_linux/GEMM_CUDA$ echo 'liam ALL=(ALL) NOPASSWD: /usr/local/cuda/bin/ncu' | sudo tee /etc/sudoers.d/ncu
#      liam ALL=(ALL) NOPASSWD: /usr/local/cuda/bin/ncu

#      echo 'liam ALL=(ALL) NOPASSWD: /usr/local/cuda/bin/ncu'
#
#      - 输出一行 sudoers 规则到 stdout
#      - 单引号：内容原样输出，不解析变量和特殊字符
#      - 规则含义：
#        - liam：适用的用户
#        - ALL=：在任何主机上
#        - (ALL)：可以切换为任何用户身份执行
#        - NOPASSWD:：免密
#        - /usr/local/cuda/bin/ncu：只对这一个命令免密，其他 sudo 命令不受影响
#        ● ALL=(ALL) 是 sudoers 的固定语法格式，不是可选的：
#          - host（第一个ALL）：指定这条规则在哪台主机上生效
#            - ALL = 所有主机，单机环境写 ALL 是惯例
#            - 也可以写具体主机名，用于多机共享同一 sudoers 配置的场景
#          - runas（括号里的ALL）：指定可以切换为哪个用户身份来执行命令
#            - (ALL) = 可以切换为任何用户（包括 root）
#            - 也可以写 (root) 限定只能以 root 身份执行

#      - 管道|，把左侧命令的 stdout 传给右侧命令的 stdin

#      sudo tee /etc/sudoers.d/ncu
#
#      - tee：从 stdin 读取内容，同时写入文件和 stdout
#      - sudo：以 root 权限写文件（/etc/sudoers.d/ 普通用户无写权限）
#      - /etc/sudoers.d/ncu：sudoers 的附属配置目录，系统会自动加载其中的文件，比直接修改 /etc/sudoers 更安全
#
#      为什么不用 sudo echo ... >> /etc/sudoers.d/ncu
#
#      sudo echo '...' >> /etc/sudoers.d/ncu  # ✗ 错误
#      原因：>> 重定向由当前 shell 处理，shell 在执行命令前先尝试以当前用户打开文件
#        执行顺序：
#          1. shell 以 liam 身份尝试打开 /etc/sudoers.d/ncu 用于写入
#          2. liam 无写权限 → shell 直接报 Permission denied，命令未执行
#          3. sudo echo '...' 根本没有机会运行
#        sudo 只提升了 echo 的权限，>> 归 shell 管，shell 本身是普通用户权限
#
#      sudo tee 为什么能工作：
#        echo '...' | sudo tee /etc/sudoers.d/ncu
#          1. shell 执行 echo，输出到管道（不涉及文件权限）
#          2. sudo tee 以 root 身份启动，由 tee 自己打开并写入文件
#          3. 文件的打开操作由 root 执行 → 有权限 ✓
#        tee 是以 root 身份运行的进程，绕开了 shell 的权限限制
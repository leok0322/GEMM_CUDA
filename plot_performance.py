#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
# 上方 `# /// script` 块是 uv 的 inline script metadata，
# 让 `uv run` 自动安装依赖；直接用 python 运行时该块无效，需手动 pip install matplotlib
"""
用法：
  uv run plot_performance.py 512 1024 2048 4096
  uv run plot_performance.py --all
  uv run plot_performance.py            # 默认绘制全部维度
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# benchmark_results/ 与本脚本同目录，用 __file__ 保证无论从哪里调用路径都正确
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
ALL_DIMS = [128, 256, 512, 1024, 2048, 4096]
KERNELS = list(range(0, 11))  # kernel 0（cuBLAS）到 kernel 10

KERNEL_LABELS = {
    0:  "K0  cuBLAS",
    1:  "K1  Naive",
    2:  "K2  Coalescing",
    3:  "K3  SMEM Block",
    4:  "K4  Regs 1D",
    5:  "K5  Regs 2D",
    6:  "K6  Vec+As Col",
    7:  "K7  Bs Rearrange",
    8:  "K8  Bs Padding",
    9:  "K9  Autotuned",
    10: "K10 WarpTiling",
}


def parse_results(kernel_id: int) -> dict[int, float]:
    """解析 kernel_{id}_result.txt，返回 {dimension: gflops}。
    返回示例：{128: 45.2, 256: 123.4, 512: 456.7, 1024: 890.1, 2048: 1234.5, 4096: 2345.6}
    """
    path = RESULTS_DIR / f"kernel_{kernel_id}_result.txt"
    if not path.exists():
        return {}
    # 匹配形如 "performance: ( 1234.56 ) GFLOPS. size: (2048)" 的行
    # group(1) → GFLOPS（float），group(2) → 矩阵维度（int，作为 key）
    pattern = re.compile(
        r"performance:\s*\(\s*([\d.]+)\s*\)\s*GFLOPS\.\s*size:\s*\((\d+)\)"
    )
    results = {}
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            gflops, size = float(m.group(1)), int(m.group(2))
            results[size] = gflops  # 同一 size 多次出现时取最后一行
    return results


def main():
    parser = argparse.ArgumentParser(description="绘制各 kernel 在指定维度上的 GFLOPS 折线图")
    parser.add_argument(
        "dims",
        nargs="*",   # 0 个或多个位置参数
        type=int,
        metavar="DIM",
        help=f"要绘制的矩阵维度，可选值：{ALL_DIMS}（不填则使用全部维度）",
    )
    # action="store_true"：传了 --all 则 args.all=True，不传则 False
    parser.add_argument("--all", action="store_true", help="使用全部维度（与不填参数等价）")
    # 解析 sys.argv[1:]，结果存入 args 命名空间（args.dims, args.all）
    args = parser.parse_args()

    # args.dims 非空 且 未传 --all → 对用户输入去重(set)并升序排列(sorted)
    # 否则（未传维度 或 传了 --all）→ 使用全部维度；--all 优先级高于位置参数
    dims = sorted(set(args.dims)) if args.dims and not args.all else ALL_DIMS

    # 拒绝结果文件中不存在的维度，避免绘出空点
    invalid = [d for d in dims if d not in ALL_DIMS]
    if invalid:
        print(f"错误：不支持的维度 {invalid}，可选值为 {ALL_DIMS}", file=sys.stderr)
        sys.exit(1)

    # 读取所有 kernel 数据，缺失文件只警告不退出
    data: dict[int, dict[int, float]] = {}
    for kid in KERNELS:
        parsed = parse_results(kid)
        if parsed:
            data[kid] = parsed
        else:
            print(f"警告：未找到 kernel {kid} 的结果文件，跳过", file=sys.stderr)

    if not data:
        print("错误：没有找到任何结果文件", file=sys.stderr)
        sys.exit(1)

    # X 轴为 kernel 1-10，每条折线代表一个矩阵维度
    plot_kernels = list(range(1, 11))
    # tab10 调色板提供 10 种区分度高的颜色，每个维度一种
    colors = [plt.cm.tab10(i) for i in range(len(dims))]

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, dim in enumerate(dims):
        color = colors[i]
        ys = [data.get(kid, {}).get(dim) for kid in plot_kernels]
        # 过滤掉 None（该 kernel 无此维度数据），折线在缺失处自动断开
        valid = [(kid, v) for kid, v in zip(plot_kernels, ys) if v is not None]
        if not valid:
            continue
        xs, ys_valid = zip(*valid)
        ax.plot(xs, ys_valid, label=f"{dim}×{dim}", marker="o",
                linewidth=1.8, markersize=5, color=color)

        # cuBLAS 基准：与该维度同色的水平虚线，方便直接对比每个 kernel 与 cuBLAS 的差距
        cublas = data.get(0, {}).get(dim)
        if cublas is not None:
            ax.axhline(cublas, linestyle="--", linewidth=1.0, color=color, alpha=0.5)

    ax.set_xlabel("Kernel", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=12)
    ax.set_title("CUDA GEMM Kernel Performance", fontsize=14)
    # X 轴刻度：kernel 编号，标签使用简短名称，旋转避免重叠
    ax.set_xticks(plot_kernels)
    ax.set_xticklabels([KERNEL_LABELS[k] for k in plot_kernels], rotation=25, ha="right")
    # Y 轴千位分隔符，方便读大数值
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    # 图例：实线=各维度折线；在图例末尾加一条灰色虚线说明 cuBLAS 基准含义
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], linestyle="--", linewidth=1.0, color="gray", alpha=0.7))
    labels.append("cuBLAS baseline")
    ax.legend(handles, labels, loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)  # Y 轴从 0 开始，避免视觉上夸大差距

    plt.tight_layout()
    out = Path(__file__).parent / "performance_plot.png"
    plt.savefig(out, dpi=150)
    print(f"已保存：{out}")

    # 用图片相对路径替换 README.md 中 <!-- performance_plot --> 标记之间的内容
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        img_rel = out.relative_to(Path(__file__).parent)  # 相对于 README 所在目录的路径
        new_block = f"<!-- performance_plot -->\n![CUDA GEMM Kernel Performance]({img_rel})\n<!-- performance_plot -->"
        updated = re.sub(
            r"<!-- performance_plot -->.*<!-- performance_plot -->",
            new_block,
            readme.read_text(),  # 传入完整 README 文本作为匹配目标
            flags=re.DOTALL,     # 默认 . 不匹配 \n；re.DOTALL 让 . 匹配包括换行在内的任意字符，从而跨行匹配两个标记之间的旧内容
        )
        readme.write_text(updated)
        print(f"已更新：{readme}")

    plt.show()


if __name__ == "__main__":
    main()

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

    fig, ax = plt.subplots(figsize=(12, 7))

    for kid, results in data.items():
        # results.get(d) 在该 kernel 没有某维度数据时返回 None
        y = [results.get(d) for d in dims]
        # 过滤掉 None，只绘制有数据的点；某 kernel 在部分维度缺测时折线自动断开
        valid = [(d, v) for d, v in zip(dims, y) if v is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)

        style = dict(marker="o", linewidth=1.8, markersize=5)
        # cuBLAS（kid=0）用虚线灰色加粗，作为性能上限参考基线
        if kid == 0:
            style.update(linewidth=2.2, linestyle="--", color="gray", markersize=6)

        ax.plot(xs, ys, label=KERNEL_LABELS[kid], **style)

    ax.set_xlabel("Matrix Dimension (m=n=k)", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=12)
    ax.set_title("CUDA GEMM Kernel Performance", fontsize=14)
    # X 轴只在实际维度处打刻度，避免线性插值出现无意义的中间值
    ax.set_xticks(dims)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    # Y 轴千位分隔符，方便读大数值
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)  # Y 轴从 0 开始，避免视觉上夸大差距

    plt.tight_layout()
    out = Path(__file__).parent / "benchmark_results" / "performance_plot.png"
    plt.savefig(out, dpi=150)
    print(f"已保存：{out}")
    plt.show()


if __name__ == "__main__":
    main()

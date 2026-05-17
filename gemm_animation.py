"""
GEMM 动画 —— 逐元素可视化矩阵乘法 C = A × B。

渲染命令：
    uv run python -m manim -pql gemm_animation.py GEMMScene   # 低画质，渲染完自动预览
    uv run python -m manim -pqm gemm_animation.py GEMMScene   # 中画质
    uv run python -m manim -pqh gemm_animation.py GEMMScene   # 高画质（较慢）
"""
from manim import *
import numpy as np

ROW_COLOR    = BLUE_C
COL_COLOR    = GREEN_C
ACTIVE_COLOR = YELLOW
DONE_COLOR   = TEAL_C


class GEMMScene(Scene):
    """动画演示通用矩阵乘法：C = A × B（3×3 整数示例）。"""

    A_DATA = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]

    B_DATA = [[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]]

    # 3 格 × 0.72 = 2.16 单位；五组元素（A、×、B、=、C）加上 0.5 间距
    # 总宽约 13 单位，略小于 Manim 默认帧宽 14.2 单位，恰好不越界。
    CS = 0.72

    def construct(self):
        A_np   = np.array(self.A_DATA)
        B_np   = np.array(self.B_DATA)
        # .tolist() 将 numpy 整数类型转为原生 Python int，
        # 避免后续 str() / int() 因 dtype 不同产生意外行为。
        C_list = (A_np @ B_np).tolist()
        n      = 3

        # ── 1. 标题 ──────────────────────────────────────────────────────────
        title = Text("GEMM:  C = A × B", font_size=40)
        self.play(Write(title))
        self.wait(0.8)
        self.play(title.animate.to_edge(UP))

        # ── 2. 构建矩阵网格 ───────────────────────────────────────────────────
        A_grid, A_sq, A_tx = self._make_grid(self.A_DATA)
        B_grid, B_sq, B_tx = self._make_grid(self.B_DATA)
        C_grid, C_sq, C_tx = self._make_grid([["?"] * n for _ in range(n)])

        lA = Text("A", font_size=26, color=ROW_COLOR).next_to(A_grid, UP, buff=0.15)
        lB = Text("B", font_size=26, color=COL_COLOR).next_to(B_grid, UP, buff=0.15)
        lC = Text("C", font_size=26, color=ACTIVE_COLOR).next_to(C_grid, UP, buff=0.15)

        times_sym = Text("×", font_size=34)
        eq_sym    = Text("=", font_size=34)

        # 水平排列：[A]  ×  [B]  =  [C]
        # 外层 VGroup 故意不赋变量：arrange() 直接修改子对象的位置，
        # A_sq / B_sq / C_sq 的引用在调用后仍然有效，无需保留外层包装。
        VGroup(
            VGroup(A_grid, lA), times_sym,
            VGroup(B_grid, lB), eq_sym,
            VGroup(C_grid, lC),
        ).arrange(RIGHT, buff=0.5).center()

        self.play(
            FadeIn(VGroup(A_grid, lA)),
            FadeIn(VGroup(B_grid, lB)),
            FadeIn(VGroup(C_grid, lC)),
            Write(times_sym),
            Write(eq_sym),
        )
        self.wait(0.5)

        # ── 3. 逐元素计算点积，填入 C[i][j] ──────────────────────────────────
        for i in range(n):
            for j in range(n):
                # 高亮 A 的第 i 行（蓝色）和 B 的第 j 列（绿色）
                self.play(
                    *[A_sq[i][k].animate.set_fill(ROW_COLOR, opacity=0.6)
                      for k in range(n)],
                    *[B_sq[k][j].animate.set_fill(COL_COLOR, opacity=0.6)
                      for k in range(n)],
                    C_sq[i][j].animate.set_fill(ACTIVE_COLOR, opacity=0.6),
                    run_time=0.45,
                )

                result = int(C_list[i][j])
                terms  = " + ".join(
                    f"{self.A_DATA[i][k]}·{self.B_DATA[k][j]}"
                    for k in range(n)
                )
                formula = Text(f"{terms} = {result}", font_size=20)
                formula.to_edge(DOWN, buff=0.4)
                self.play(Write(formula), run_time=0.4)
                self.wait(0.2)

                # new_lbl 默认创建在原点；必须先调用 move_to，
                # 否则 Transform 动画会终止于 (0, 0) 而非目标格子中心。
                new_lbl = Text(str(result), font_size=20)
                new_lbl.move_to(C_tx[i][j].get_center())
                self.play(
                    Transform(C_tx[i][j], new_lbl),
                    C_sq[i][j].animate.set_fill(DONE_COLOR, opacity=0.4),
                    run_time=0.4,
                )

                # opacity=0 才是清除填充色的关键；BLACK 是必填的位置参数，
                # 但 opacity 为零时颜色值本身无意义。
                self.play(
                    *[A_sq[i][k].animate.set_fill(BLACK, opacity=0) for k in range(n)],
                    *[B_sq[k][j].animate.set_fill(BLACK, opacity=0) for k in range(n)],
                    FadeOut(formula),
                    run_time=0.3,
                )

        # ── 4. 结束 ──────────────────────────────────────────────────────────
        done = Text("Matrix Multiplication Complete!", font_size=30, color=DONE_COLOR)
        self.play(Write(done.to_edge(DOWN, buff=0.4)))
        self.wait(2.0)

    def _make_grid(self, data):
        """构建矩阵网格 VGroup，返回 (group, squares[][], texts[][])。"""
        n_rows, n_cols = len(data), len(data[0])
        group, squares, texts = VGroup(), [], []
        for i in range(n_rows):
            sq_row, tx_row = [], []
            for j in range(n_cols):
                sq = Square(side_length=self.CS, color=WHITE, stroke_width=1.5)
                sq.move_to(RIGHT * j * self.CS + DOWN * i * self.CS)
                tx = Text(str(data[i][j]), font_size=20).move_to(sq.get_center())
                group.add(sq, tx)
                sq_row.append(sq)
                tx_row.append(tx)
            squares.append(sq_row)
            texts.append(tx_row)
        # 将网格居中，使其包围盒关于原点对称。
        # arrange() 以包围盒中点为基准移动整组；若不提前居中，
        # 标签与矩阵之间会产生偏移错位。
        group.center()
        return group, squares, texts


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run(
        [
            sys.executable,  # 当前 Python 解释器路径，确保与脚本使用同一环境
            "-m", "manim",   # 以模块方式运行 manim，等价于命令行 python -m manim
            "-pql",          # -p 渲染后自动预览；-q 指定画质；l 为 low 低画质（最快）
            __file__,        # 当前脚本路径，告知 manim 从哪个文件查找场景
            "GEMMScene",     # 要渲染的场景类名
        ],
        check=True,  # 渲染失败时抛出异常，而非静默退出
    )

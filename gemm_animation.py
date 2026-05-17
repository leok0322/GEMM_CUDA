"""
GEMM 动画 —— 逐元素可视化矩阵乘法 C = A × B。

渲染命令：
    uv run python -m manim -pql gemm_animation.py GEMMScene   # 低画质，渲染完自动预览
    uv run python -m manim -pqm gemm_animation.py GEMMScene   # 中画质
    uv run python -m manim -pqh gemm_animation.py GEMMScene   # 高画质（较慢）
"""
from manim import *
import numpy as np

# partial / 子类遮蔽均无效：Manim 在渲染时通过 manim 模块直接引用原始 Text，
# 模块级的名字遮蔽对其不可见。唯一可靠的方案是在每个含中文的 Text 调用处
# 显式传入 font 参数。
# 依赖：sudo apt-get install fonts-noto-cjk
CJK = "Noto Sans CJK SC"   # 含中文的 Text(...) 调用统一用 font=CJK

ROW_COLOR    = BLUE_C
COL_COLOR    = GREEN_C
ACTIVE_COLOR = YELLOW
DONE_COLOR   = TEAL_C
SMEM_COLOR   = ORANGE   # shared memory 高亮色
REG_COLOR    = RED_C    # 寄存器加载高亮色


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


class Kernel10Scene(Scene):
    """
    Kernel 10 (gemmWarptiling_v2)：Warp Tiling + 寄存器复用动画。

    以 Thread 0 为代表演示核心改进：WM 与线程布局解耦后 threadRowIterNum > 1，
    每次 innerCol 迭代只从 SMEM 各读一次，存入寄存器后做外积，
    regAsCache 每个元素复用 TN 次，regBsCache 每个元素复用 threadRowIterNum×TM 次。

    演示参数：
      BM=BN=4, BK=2, WM=WN=4（整块为单一 warp tile）
      TM=1, TN=2, NUM_THREADS=4
      threadRowIterNum = WM / (NUM_THREADS/(WN/TN)*TM) = 4/(4/2*1) = 2
      Thread 0：rowGroup=0, colGroup=0 → rows[0,1], cols[0,1]
      每次 innerCol：4 次 FMA，仅 2+2=4 次 SMEM 读
    """

    BM = BN = 4
    BK = 2
    TM = 1
    TN = 2
    NUM_THREADS = 4
    CS = 0.62   # 单元格边长

    # 演示用矩阵数据（As: BM×BK，Bs: BK×BN）
    AS_DATA = [[1, 2], [3, 4], [5, 6], [7, 8]]
    BS_DATA = [[1, 2, 3, 4], [5, 6, 7, 8]]

    def construct(self):
        # ── 派生计算参数 ───────────────────────────────────────────────────────
        wn_tn = self.BN // self.TN                       # 列组数 = WN/TN = 2
        trni  = self.NUM_THREADS // wn_tn * self.TM      # threadrowNumPerIter = 2
        tri   = self.BM // trni                          # threadRowIterNum = 2
        # Thread 0 负责的行列范围
        t0_rows = list(range(tri * self.TM))              # [0, 1]
        t0_cols = list(range(self.TN))                    # [0, 1]

        # 独立追踪 C 的数值，与 Manim Text 对象同步，避免从渲染对象反向读取数字
        c_vals = [[0] * self.BN for _ in range(self.BM)]

        # ── 1. 标题 ────────────────────────────────────────────────────────────
        title = Text("Kernel 10: Warp Tiling + Register Reuse", font_size=34)
        self.play(Write(title))
        self.wait(0.8)
        self.play(title.animate.to_edge(UP))

        # ── 2. 参数说明 ────────────────────────────────────────────────────────
        param_lbl = Text(
            f"BM=BN={self.BM}  BK={self.BK}  TM={self.TM}  TN={self.TN}"
            f"  threadRowIterNum={tri}  ->  each thread: {tri} rows x {self.TN} cols",
            font_size=17, color=GRAY,
        ).next_to(title, DOWN, buff=0.1)
        self.play(FadeIn(param_lbl))

        # ── 3. 构建矩阵网格 ────────────────────────────────────────────────────
        As_grid, As_sq, As_tx = self._make_grid(self.AS_DATA)
        Bs_grid, Bs_sq, Bs_tx = self._make_grid(self.BS_DATA)
        C_grid,  C_sq,  C_tx  = self._make_grid([["0"] * self.BN for _ in range(self.BM)])

        lAs = Text("As  (Shared Mem  BM×BK=4×2)", font_size=14, color=SMEM_COLOR).next_to(As_grid, UP, buff=0.1)
        lBs = Text("Bs  (Shared Mem  BK×BN=2×4)", font_size=14, color=SMEM_COLOR).next_to(Bs_grid, UP, buff=0.1)
        lC  = Text("C   (Result  BM×BN=4×4)",      font_size=14, color=DONE_COLOR).next_to(C_grid,  UP, buff=0.1)

        # 经典排列：As 左，Bs 右上，C 右下
        # As 与 C 共享 y 中心（行对齐）；Bs 与 C 共享 x 中心（列对齐）
        VGroup(As_grid, lAs).move_to(LEFT * 3.2 + DOWN * 0.3)
        VGroup(Bs_grid, lBs).move_to(RIGHT * 1.3 + UP  * 1.9)
        VGroup(C_grid,  lC ).move_to(RIGHT * 1.3 + DOWN * 0.3)

        self.play(
            FadeIn(VGroup(As_grid, lAs)),
            FadeIn(VGroup(Bs_grid, lBs)),
            FadeIn(VGroup(C_grid,  lC)),
        )
        self.wait(0.4)

        # ── 4. 沿 K 方向逐 innerCol 迭代（共 BK=2 次）────────────────────────
        for bk in range(self.BK):
            bk_lbl = Text(f"innerCol = {bk}", font_size=20).to_edge(DOWN, buff=0.85)
            self.play(Write(bk_lbl), run_time=0.3)

            # ── 4a. 高亮 As 第 bk 列和 Bs 第 bk 行（整块 SMEM 可用数据）────────
            self.play(
                *[As_sq[r][bk].animate.set_fill(SMEM_COLOR, opacity=0.5)
                  for r in range(self.BM)],
                *[Bs_sq[bk][c].animate.set_fill(SMEM_COLOR, opacity=0.5)
                  for c in range(self.BN)],
                run_time=0.45,
            )

            # ── 4b. Thread 0 加载寄存器（深色高亮，仅读两组元素）─────────────
            reg_as = [self.AS_DATA[r][bk] for r in t0_rows]   # regAsCache[tri*TM]
            reg_bs = [self.BS_DATA[bk][c] for c in t0_cols]   # regBsCache[TN]

            # regAsCache 来自 As 的第 bk 列中 t0_rows 对应行；
            # regBsCache 来自 Bs 的第 bk 行中 t0_cols 对应列。
            # 两组数据各只读一次，后续外积循环不再访问 SMEM。
            self.play(
                *[As_sq[r][bk].animate.set_fill(REG_COLOR, opacity=0.85)
                  for r in t0_rows],
                *[Bs_sq[bk][c].animate.set_fill(REG_COLOR, opacity=0.85)
                  for c in t0_cols],
                run_time=0.4,
            )

            reg_lbl = Text(
                f"Thread 0  regAsCache={reg_as} (reused {self.TN}x)"
                f"   regBsCache={reg_bs} (reused {tri * self.TM}x)",
                font_size=16, color=REG_COLOR,
            ).next_to(bk_lbl, UP, buff=0.12)
            self.play(Write(reg_lbl), run_time=0.4)
            self.wait(0.3)

            # ── 4c. 外积：regAsCache[ri] × regBsCache[ci] → C[r][c] ──────────
            # 外层 ri 循环：regBsCache[ci] 被复用 tri*TM 次
            # 内层 ci 循环：regAsCache[ri] 被复用 TN 次
            for ri, r in enumerate(t0_rows):
                for ci, c in enumerate(t0_cols):
                    prod = reg_as[ri] * reg_bs[ci]
                    c_vals[r][c] += prod

                    self.play(
                        C_sq[r][c].animate.set_fill(ACTIVE_COLOR, opacity=0.6),
                        run_time=0.2,
                    )
                    # move_to 必须在 Transform 之前调用，否则动画终止于原点
                    new_lbl = Text(str(c_vals[r][c]), font_size=18)
                    new_lbl.move_to(C_tx[r][c].get_center())
                    self.play(
                        Transform(C_tx[r][c], new_lbl),
                        C_sq[r][c].animate.set_fill(DONE_COLOR, opacity=0.35),
                        run_time=0.25,
                    )

            # ── 4d. 清除本轮高亮 ──────────────────────────────────────────────
            self.play(
                *[As_sq[r][bk].animate.set_fill(BLACK, opacity=0)
                  for r in range(self.BM)],
                *[Bs_sq[bk][c].animate.set_fill(BLACK, opacity=0)
                  for c in range(self.BN)],
                FadeOut(bk_lbl),
                FadeOut(reg_lbl),
                run_time=0.35,
            )

        # ── 5. 结束 ────────────────────────────────────────────────────────────
        done = Text(
            "Each innerCol reads SMEM once"
            "  ->  regAsCache reused TN times, regBsCache reused TRI*TM times",
            font_size=20, color=DONE_COLOR,
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(done))
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
                tx = Text(str(data[i][j]), font_size=18).move_to(sq.get_center())
                group.add(sq, tx)
                sq_row.append(sq)
                tx_row.append(tx)
            squares.append(sq_row)
            texts.append(tx_row)
        # center() 使包围盒关于原点对称，之后 move_to 才能精确定位
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

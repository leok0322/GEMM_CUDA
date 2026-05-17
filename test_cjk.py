from manim import *

class TestCJK(Scene):
    def construct(self):
        self.add(Text("中文测试 CJK", font="Noto Sans CJK SC", font_size=40))
        self.wait()

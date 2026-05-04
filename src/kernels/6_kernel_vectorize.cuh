#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ══════════════════════════════════════════════════════════════════════════════
// kernel_6 改进：向量化内存访问（float4）
//
// ── float4 的定义 ────────────────────────────────────────────────────────────
//
//   CUDA 在 cuda_runtime.h 中定义了 float4：
//
//   struct __device_builtin__ __builtin_align__(16) float4 { float x, y, z, w; };
//   typedef __device_builtin__ struct float4 float4;
//
//   struct：4 个 float 成员连续排列，共 16 字节
//     [x: 4字节][y: 4字节][z: 4字节][w: 4字节]
//
//   __builtin_align__(16)：声明 float4 类型的对齐需求为 16 字节
//     只对编译器分配的栈/local memory 生效，寄存器无地址故对齐无意义：
//       float4 tmp;  // 编译器把 tmp 放进 4 个寄存器，__builtin_align__ 不起作用
//       &tmp;        // 取地址迫使 tmp 挪到栈/local memory，此时编译器才保证地址对齐
//     SMEM 通过 bank 系统访问，无 sector 对齐要求，__builtin_align__ 对 SMEM 无实际作用
//     对 reinterpret_cast 强转的 GMEM 指针无能为力：
//       float4 *p = reinterpret_cast<float4*>(&A[1]);  // 偏移4字节，未对齐
//       float4 tmp = p[0];  // __builtin_align__ 无法阻止，责任在程序员
//
//   typedef：C 风格类型别名，让 float4 可以不写 struct 直接使用
//     C++ 中不需要 typedef，但 CUDA 头文件为兼容 C 保留此写法
//
// ── 16 字节对齐的含义及未对齐的后果 ─────────────────────────────────────────
//
//   16 字节对齐 = 地址 % 16 == 0：
//     0x00 = 0   → 0 % 16 = 0  ✓
//     0x10 = 16  → 16 % 16 = 0 ✓
//     0x04 = 4   → 4 % 16 = 4  ✗
//
//   GPU 内存控制器以 32 字节（sector）为最小事务单元，事务起始地址必须对齐。
//   ld.global.v4.f32 取 16 字节，要求这 16 字节完整落在同一 sector 内：
//     从 0x10 取 16 字节：0x10~0x1F → 完整在一个 sector 内 ✓
//     从 0x18 取 16 字节：0x18~0x27 → 跨越 0x20 sector 边界 ✗ → 需两次事务
//
//   未对齐时的后果：
//     旧架构（Kepler 之前）：触发 misaligned address 错误，kernel 崩溃
//     新架构（Volta+）      ：硬件自动拆成多次标量加载，不崩溃但性能下降，
//                             且属于 C++ 未定义行为（reinterpret_cast 要求对齐）
//
//   对齐安全性 = 基址对齐 + 偏移量对齐，两者缺一不可：
//     cudaMalloc 返回基址   ← CUDA runtime 保证 256 字节对齐（是 16 的整数倍）
//     基址 + 偏移量         ← 程序员负责，需确保偏移也是 16 的倍数
//     栈/local memory 上的 float4 ← 编译器负责，__builtin_align__ 强制生效
//
//   本 kernel 为何安全：
//     cudaMalloc 保证基址 256 字节对齐
//     innerColA ∈ {0, 1}，偏移量 = innerColA * 4 个 float = innerColA * 16 字节
//     0×16=0，1×16=16，均是 16 的整数倍 → 基址 + 偏移量仍满足 16 字节对齐 ✓
//
// ── __device_builtin__ 的作用 ────────────────────────────────────────────────
//
//   __device_builtin__ 是条件编译宏：
//     nvcc 编译设备代码时：展开为 __attribute__((device_builtin))
//       告知编译器这是内置向量类型，映射到 PTX 的 .v4.f32 向量类型，
//       生成 ld.global.v4.f32 / st.global.v4.f32 向量指令
//     g++ 编译主机代码时：展开为空（空宏），g++ 完全感知不到它的存在
//
//   nvcc 编译 .cu 文件时分两路：
//     设备代码 → nvcc 编译 → PTX/SASS（__device_builtin__ 生效）
//     主机代码 → 交给 g++ 编译（__device_builtin__ 已被预处理器删掉）
//
//   struct 定义和 typedef 两处都写 __device_builtin__：
//     确保通过 struct float4 和 float4 两种写法都能携带该属性，
//     因为 C++ typedef 只是名字别名，属性不一定自动继承
//
// ── ld.global.v4.f32 指令解析 ────────────────────────────────────────────────
//
//   ld.global.v4.f32 是 PTX 汇编指令，各部分含义：
//     ld     : load，从内存读数据到寄存器
//     global : 地址空间为全局内存（GMEM，即显存）
//     v4     : vector 4，一次读 4 个元素
//     f32    : 每个元素是 32-bit float
//   合起来：从 GMEM 一次读 4 个 float（128-bit）到寄存器
//
//   PTX 全局内存向量加载最大支持 v4（128-bit），没有 v8：
//     ld.global.f32    r0, [addr]            // 32-bit，1 个 float
//     ld.global.v2.f32 {r0,r1}, [addr]       // 64-bit，2 个 float
//     ld.global.v4.f32 {r0,r1,r2,r3}, [addr] // 128-bit，4 个 float（上限）
//   v4 是硬件指令集的最大向量宽度，float4 正好对应这一上限
//   kernel_5 用标量加载，4 个 float 需要 4 条指令；
//   kernel_6 用 v4，一条指令完成，指令压力降为 1/4
//
// ── float4 存储位置 ──────────────────────────────────────────────────────────
//
//   float4 tmp 是函数内的局部变量，编译器拆成 4 个独立寄存器：
//     tmp.x → r0,  tmp.y → r1,  tmp.z → r2,  tmp.w → r3
//   PTX 层面：
//     ld.global.v4.f32 {r0,r1,r2,r3}, [addr]  ← 一条指令，从 GMEM 加载 16 字节
//     st.global.v4.f32 [addr], {r0,r1,r2,r3}  ← 一条指令，写回 GMEM
//   float4 是源码层面的封装，编译后该结构体不存在，只有 4 个寄存器
//   对齐要求针对 GMEM 中的源地址，寄存器没有地址概念，无对齐要求
//
// ── 为何用 float4 而不用 float arr[4] ───────────────────────────────────────
//
//   两者在 GMEM 中内存布局相同（16 字节连续），但有三个关键差异：
//
//   ① nvcc 不认识 float[4] 是向量类型
//     float4 带有 __device_builtin__ 属性，nvcc 识别后生成 ld.global.v4.f32
//     float arr[4] 没有该属性，nvcc 生成 4 条独立标量 load，向量指令优化失效
//
//   ② 对齐保证不同
//     float4：__builtin_align__(16)，16 字节对齐，满足 ld.global.v4.f32 要求
//     float arr[4]：alignof(float) = 4 字节对齐，不满足向量加载的对齐要求
//
//   ③ 数组不可直接赋值，float4 结构体可以
//     float4 a, b;  a = b;     // ✓ 结构体赋值，拷贝全部 4 个成员
//     float a[4], b[4]; a = b; // ✗ 编译报错：array type 'float[4]' is not assignable
//
//     数组退化为指针发生在右值场景（传参、算术），但赋值左侧不退化：
//       语言规定数组类型本身不可赋值，不存在"把 float* 赋给 float[4]"的操作
//       即使右侧 b 退化为 float*，也无法赋给左侧数组名（数组名不是可修改的指针变量）
//
// ── 提高 GFLOPS 的两条路径 ───────────────────────────────────────────────────
//
//   GFLOPS = FLOPs / 时间，FLOPs 固定，只能缩短时间：
//
//   路径①：提高算术强度（算术强度 = FLOPs / GMEM字节）
//     → 减少 GMEM 字节 → 让 kernel 进入 compute-bound 区 → GFLOPS 上升
//     方法：
//       a. 合并访问（提高 sector 利用率，减少无效 GMEM 字节）
//       b. 数据加载到 SMEM 复用（多线程共享同一份 GMEM 数据，GMEM 读取字节减少）
//       c. 更大的 tile（BM/BN 越大，每次 GMEM 加载被复用次数越多，AI 越高）
//     注意：已 compute-bound 时，算术强度再高也突破不了 FMA 峰值
//
//   路径②：提高执行效率（减少 FMA 单元空转）
//     → FMA 单元更少等待 → 时间缩短 → GFLOPS 上升
//     GMEM 字节不变，AI 不变，改变的是 FMA 是否在等数据
//     方法：
//       a. 消除 SMEM bank conflict（FMA 不再等 SMEM 串行化）
//       b. 数据加载到寄存器复用（regM/regN，SMEM 访问次数大幅减少，FMA 等待减少）
//       c. 减少 load 指令数，缩短加载阶段（float4 的收益）
//
//   两条路径的最终目标都是让 FMA 更满，途径不同：
//     路径①：让 kernel 进入 compute-bound 区（AI 够高，内存不再是瓶颈）
//     路径②：在 compute-bound 区内减少 FMA 空转（FMA 实际利用率接近峰值）
//
//   ┌──────────────┬──────────────────────┬──────────────┬──────────────┐
//   │ 优化手段     │ 改变的量             │ 路径         │ 效果         │
//   ├──────────────┼──────────────────────┼──────────────┼──────────────┤
//   │ SMEM 复用    │ GMEM 字节 ↓，AI ↑   │ ①           │ 进入compute  │
//   │ 扩大 block   │ GMEM 字节 ↓，AI ↑   │ ①           │ 进入compute  │
//   │ 寄存器复用   │ SMEM 访问减少        │ ②           │ FMA 空转 ↓   │
//   │ 消除bankconf │ FMA 等待减少         │ ②           │ FMA 空转 ↓   │
//   └──────────────┴──────────────────────┴──────────────┴──────────────┘
//
// ── 路径②的两种子机制：latency 隐藏 vs throughput ──────────────────────────
//
//   路径②内部有两类截然不同的机制，compute-bound 时敏感度不同：
//
//   【机制A】bank conflict → latency 问题 → 可被 warp 切换隐藏
//
//     bank conflict 让当前 warp stall，调度器切换到其他 ready warp，
//     FMA 单元继续跑其他 warp 的计算，conflict 延迟被掩盖。
//     compute-bound 时 ready warp 充足，调度器从不缺活 → conflict 影响有限。
//
//   【机制B】加载阶段指令数 → throughput 问题 → 无法隐藏
//
//     加载阶段和计算阶段被 __syncthreads__ 严格隔开，不重叠：
//
//       加载阶段：GMEM→SMEM（load/store 指令在此）
//       __syncthreads()    ← 全 block 等齐后才放行
//       计算阶段：SMEM→寄存器→FMA
//
//     store 指令只在加载阶段执行，FMA 只在计算阶段执行，
//     两者不同时竞争发射槽，所以问题不是"store 抢了 FMA 的发射槽"。
//
//     真正的损失：加载阶段指令多 → 加载阶段耗时长 → sync 屏障推迟 → FMA 等待：
//
//       指令少：[──加载──]──sync──[────FMA────]
//       指令多：[────加载────]──sync──[────FMA────]
//                    ↑ 这段时间 FMA 等待（SM 上有其他 block 可部分填补，但本 block 吞吐下降）
//
//     每条指令都必须执行，无法跳过 → 加载阶段时间是硬性下限。
//
//   compute-bound 时两种机制的对比：
//     机制A（conflict）：warp 切换可隐藏延迟 → 影响较小
//     机制B（指令数） ：sync 推迟，FMA 等待 → 影响更直接
//
//   实例（kernel_6 vs kernel_7）：
//     kernel_7 把 Bs 写入从 1 条 float4 store 改为 4 条 scalar store，
//     消除了读 conflict（机制A收益），但增加了加载阶段指令数（机制B损失），
//     净效果：机制B损失 > 机制A收益 → 4377 GFLOPS < kernel_6 的 4699 GFLOPS
//
// ── FMA 与 compute-bound 的关系 ──────────────────────────────────────────────
//
//   FMA（Fused Multiply-Add）：a = b * c + d，一条指令完成乘加，计 2 个 FLOP
//
//   峰值算力的来源：
//     峰值 GFLOPS = SM数 × 每SM每周期FMA数 × 2(乘+加) × 频率
//
//     频率 = GPU 核心时钟频率（Core Clock），单位 Hz，即每秒的时钟周期数
//       每个时钟周期，每个 SM 的 FMA 单元执行一批 FMA 操作
//       频率越高，每秒能执行的 FMA 次数越多 → 峰值算力越高
//       GPU 有两种频率：
//         Base Clock ：保证能长期维持的基准频率（保守值）
//         Boost Clock：负载低、温度低时自动超频到的最高频率
//       峰值算力通常以 Boost Clock 计算
//
//     以 A100 SXM 为例：
//       SM 数          = 108
//       每SM每周期FMA数 = 64（每 SM 有 64 个 FP32 CUDA Core，每核每周期 1 FMA）
//       Boost Clock    = 1.41 GHz = 1.41 × 10⁹ Hz
//       峰值 FP32      = 108 × 64 × 2 × 1.41×10⁹ = 19.49 TFLOPS ≈ 19.5 TFLOPS ✓
//
//     FMA 吞吐 = 峰值算力，两者等价
//     → "达到峰值算力" = "FMA 单元每个周期都在执行，没有空转"
//
//   compute-bound 的定义：
//     FMA 单元是瓶颈，内存带宽不是限制因素
//     注意：compute-bound ≠ FMA 饱和；FMA 仍可因 bank conflict 等原因空转，
//           实际 GFLOPS 可能远低于峰值
//
//   compute-bound 的判断（屋顶线模型）：
//     算术强度 > Ridge Point = 峰值算力 / 内存带宽
//     → 内存带宽能喂饱的算力 > 硬件峰值 → FMA 是瓶颈 → compute-bound
//
//   compute-bound 时能做什么：
//     ① 提高算术强度已无意义（瓶颈在计算侧，减少内存访问不影响总时间）
//     ② 只能提高执行效率：减少 FMA 空转（bank conflict、指令发射压力）
//     ③ 或换用更高算力的硬件指令（Tensor Core，专用矩阵乘单元）
//
// ── 屋顶线模型（Roofline）与 compute-bound 推导 ──────────────────────────────
//
//   GFLOPS_actual = min(峰值算力, 算术强度 × 内存带宽)
//   Ridge Point   = 峰值算力 / 内存带宽   （compute-bound 与 memory-bound 的分界）
//   算术强度 > Ridge Point → compute-bound（瓶颈在计算侧，不在内存侧）
//
//   第一步：计算 kernel 的算术强度（BM=BN=128, BK=8）
//     FLOPs     = 2 × BM × BN × BK = 2 × 128 × 128 × 8 = 262144
//     GMEM 读取 = (BM×BK + BK×BN) × 4字节 = 8192 字节
//     算术强度  = 262144 / 8192 = 32 FLOPs/Byte
//
//   第二步：计算 Ridge Point（以 A100 为例）
//     峰值 FP32 算力 = 19500 GFLOPS
//     HBM 带宽       = 2000 GB/s
//     Ridge Point    = 19500 / 2000 = 9.75 FLOPs/Byte
//
//   第三步：对比
//     算术强度（32）> Ridge Point（9.75）
//     → 带宽能喂饱的算力上限 = 32 × 2000 = 64 TFLOPS > 峰值 19.5 TFLOPS
//     → 瓶颈在计算侧（compute-bound），GFLOPS 上限 = 19.5 TFLOPS
//
//   第四步：时间验证（per SM）
//     T_compute = 262144 / (19500/108 × 10⁹) = 1.45 μs
//     T_mem     = 8192   / (2000/108  × 10⁹) = 0.44 μs
//     T_compute > T_mem → 时间由计算决定 ✓
//
//   屋顶线图：
//     GFLOPS
//       ↑
//     19.5T|━━━━━━━━━━━━━━━━━━━━━  ← FMA 峰值（compute roof，上限）
//          |            ╱  ★ kernel（32 FLOPs/Byte，compute-bound 区）
//          |          ╱         但实际 GFLOPS 可能远低于 19.5T（FMA 有空转）
//          |        ╱  ← memory-bound 区
//          |      ╱
//          |╱
//          +─────────────────────→ 算术强度（FLOPs/Byte）
//                   9.75
//                   ↑ Ridge Point
//
//   屋顶线的局限：
//     只判断瓶颈在哪一侧，不告诉你 FMA 实际利用率
//     compute-bound + bank conflict 严重 → FMA 利用率 60%，实际 GFLOPS 远低于峰值
//     compute-bound + 无 bank conflict   → FMA 利用率 90%+，接近峰值
//     → 执行效率优化（消除 bank conflict）在 compute-bound 时仍能显著提高 GFLOPS
//
//   实际验证：ncu 查看 sm__pipe_fma_cycles_active 利用率是否接近 100%
//
// ── 算术强度的上限因素 ───────────────────────────────────────────────────────
//
//   算术强度 = FLOPs / GMEM字节，以下因素限制它进一步提升：
//     SMEM 容量     ：限制 tile 大小，tile 越大每次 GMEM 加载复用越多
//     寄存器数量    ：限制寄存器能缓存的数据量（TM×TN 受寄存器上限约束）
//     算法本身      ：elementwise 等算法天然 FLOPs 少，算术强度上限低
//     GMEM 访问模式 ：非合并访问浪费带宽，等效于分母（GMEM字节）虚增
//
// ── float4 收益分析 ──────────────────────────────────────────────────────────
//
//   float4 的收益属于路径②（执行效率优化）：
//     kernel_5 用标量加载 + loadOffset 循环：4 条 load + 循环控制指令/次
//     kernel_6 用 float4：1 条 load/次，消除循环，load 指令数降为 1/4
//     注：float4 不提高带宽利用率（sector 利用率两者都是 100%）
//
//   float4 能否提高 GFLOPS，取决于 FMA 是否有空转：
//
//   情况①：无 bank conflict，warp 充足，FMA 接近饱和
//     加载阶段缩短，但 compute 阶段才是瓶颈，总时间几乎不变
//     → 收益接近 0
//
//   情况②：有 bank conflict，FMA 因等 SMEM 数据而空转（本 kernel 实际情况）
//     bank conflict 让 warp 停住 → 调度器找不到就绪 FMA → 发射槽空转
//     float4 的间接收益：
//       总 tile 时间 = 加载阶段 + 计算阶段（__syncthreads__ 隔开，顺序执行）
//       load 指令少 → 加载阶段时间缩短 → 总时间缩短 → GFLOPS 上升
//     → float4 确实能提高 GFLOPS，但收益是间接且有限的
//     → 决定 FMA 利用率上限的是 bank conflict，float4 只在此基础上做边际优化
//     → 真正大收益是 As 转置消除 bank conflict（路径②的 a）
//
// ── float4 向量加载的合并访问事务分析（A 矩阵，BK=8）────────────────────────
//
//   kernel_6 加载 A 时（256 线程，BK/4=2）：
//     innerRowA = threadIdx.x / 2  → 每 2 个线程共享同一行（0..127）
//     innerColA = threadIdx.x % 2  → 0 或 1，每行 2 个 float4 位置
//
//   warp 前 8 个线程（4 对）的访问地址（K=4096）：
//     Thread 0,1 → row 0：A[0*K+0..7]   字节偏移 0
//     Thread 2,3 → row 1：A[1*K+0..7]   字节偏移 16384
//     Thread 4,5 → row 2：A[2*K+0..7]   字节偏移 32768
//     Thread 6,7 → row 3：A[3*K+0..7]   字节偏移 49152
//
//   每对线程访问同一行的连续 32 字节（1 个 sector），sector 内全部使用：
//     sector 利用率 = 32/32 = 100% ✓
//
//   但 4 对分属 4 条不同的 cache line（行间距 16384 字节 >> 128 字节）：
//     cache line 利用率 = 32/128 = 25% ✗
//
//   对比 kernel_5（每线程取 1 个 float，8 个线程一组）：
//     ┌──────────────────┬──────────────────┬──────────────────┐
//     │      粒度        │    kernel_5      │    kernel_6      │
//     ├──────────────────┼──────────────────┼──────────────────┤
//     │ 每组/对线程使用  │ 32字节（100%）   │ 32字节（100%）   │
//     │ cache line 利用率│ 32/128 = 25%     │ 32/128 = 25%     │
//     │ 每线程 load 指令 │ 1条（scalar）    │ 1条（float4）    │
//     │ 每线程加载量     │ 1 float = 4字节  │ 4 float = 16字节 │
//     │ 总 load 轮次     │ 4轮循环          │ 1轮（无循环）    │
//     └──────────────────┴──────────────────┴──────────────────┘
//
//   结论：
//     两者 cache line 利用率相同（均 25%），A 的行间距 >> cache line 无法改变
//     float4 的收益在于：消除 loadOffset 循环，load 指令总数从 4 条降为 1 条
//     带宽效率不变，指令发射压力降低
//
// ── As 转置如何消除 SMEM bank conflict ───────────────────────────────────────
//
//   SMEM 有 32 个 bank，每 bank 宽 4 字节，元素 i 的 bank = i % 32
//   同一 warp 内多个线程访问同一 bank 的不同地址 → bank conflict → 串行化
//
//   参数：BM=128, BK=8, TM=8, TN=8, BN/TN=16
//     threadRow = threadIdx.x / 16（warp 内：threads 0-15 → threadRow=0，16-31 → threadRow=1）
//     warp 内一次 SMEM 读：所有 32 线程同时发出，固定 i 和 dotIdx
//
//   kernel_5：As 行主序，As[row * BK + col]
//     读：As[(threadRow * TM + i) * BK + dotIdx]
//     元素下标：
//       threadRow=0：(0*8+i)*8+dotIdx = i*8+dotIdx
//       threadRow=1：(1*8+i)*8+dotIdx = 64+i*8+dotIdx
//     bank：
//       threadRow=0：(i*8+dotIdx) % 32
//       threadRow=1：(64+i*8+dotIdx) % 32 = (i*8+dotIdx) % 32  ← 64%32=0，bank 相同！
//     → 2 组线程访问同一 bank 的不同地址 → 2-way bank conflict → 串行化 ✗
//
//   kernel_6：As 列主序（转置），As[col * BM + row]
//     读：As[dotIdx * BM + threadRow * TM + i]
//     元素下标：
//       threadRow=0：dotIdx*128+0*8+i = dotIdx*128+i
//       threadRow=1：dotIdx*128+1*8+i = dotIdx*128+8+i
//     bank：
//       threadRow=0：(dotIdx*128+i) % 32 = i        （128%32=0）
//       threadRow=1：(dotIdx*128+8+i) % 32 = 8+i
//     → bank i ≠ bank 8+i → 两组线程访问不同 bank → 无冲突 ✓
//
//   根本原因：
//     行主序时，相邻 threadRow（间距 TM*BK=64 个元素）落在同一 bank（64%32=0）
//     列主序时，相邻 threadRow（间距 TM=8 个元素）bank 相差 8，必然不同 bank
//
// ── As 转置写入的 bank conflict 及 float4 的影响 ──────────────────────────────
//
//   转置写入（GMEM→SMEM）：
//     As[(innerColA*4+k)*BM + innerRowA]，k=0..3（对应 tmp.x/.y/.z/.w）
//     innerColA = threadIdx.x % (BK/4) = threadIdx.x % 2  → 取值 0 或 1
//     innerRowA = threadIdx.x / (BK/4) = threadIdx.x / 2
//
//   warp 内两组线程写入 k=0 时的 bank（以 innerRowA=0 为例）：
//     innerColA=0：As[0*512 + 0] = As[0],   bank = 0
//     innerColA=1：As[1*512 + 0] = As[512], bank = 512 % 32 = 0  （512=16×32）
//     → 不同地址，相同 bank → 2-way 写冲突
//
//   根本原因：列主序使相邻 innerColA 组在 SMEM 中的步长 = 4×BM = 4×128 = 512，
//             是 32 的整数倍 → 必然落入同一 bank
//
//   float4 对写冲突的影响：
//     float4 一次加载 4 列，innerColA 的取值数 = BK/4（而非 BK）
//
//     ┌─────────────────┬──────────────────┬──────────┬──────────────┐
//     │ 加载方式        │ innerColA 取值数 │ SMEM步长 │ 写 bank 冲突 │
//     ├─────────────────┼──────────────────┼──────────┼──────────────┤
//     │ 标量（无float4）│      BK=8        │ 1×BM=128 │   8-way      │
//     │ float4          │      BK/4=2      │ 4×BM=512 │   2-way      │
//     └─────────────────┴──────────────────┴──────────┴──────────────┘
//
//     两者步长都是 32 的整数倍（128=4×32，512=16×32），冲突不可避免；
//     float4 把冲突组数从 8 降到 2，减轻程度但不消除。
//
//   转置对读写 bank conflict 的综合对比：
//
//   ┌──────────────┬──────────────────────────────┬──────────────────────────────┐
//   │              │          读取（计算阶段）     │          写入（加载阶段）     │
//   ├──────────────┼──────────────────────────────┼──────────────────────────────┤
//   │ kernel_5     │ 2-way 冲突                   │ 无冲突                       │
//   │ （行主序）   │ threadRow 间距=64=2×32，      │ element index = threadIdx.x  │
//   │              │ 两组线程落在同一 bank         │ 线性写入，各线程不同 bank    │
//   ├──────────────┼──────────────────────────────┼──────────────────────────────┤
//   │ kernel_6     │ 无冲突                       │ 2-way 冲突（float4）         │
//   │ （列主序）   │ threadRow 间距=8，bank 差 8  │ innerColA 步长=512=16×32，   │
//   │              │ 两组线程落在不同 bank         │ 两组线程落在同一 bank        │
//   └──────────────┴──────────────────────────────┴──────────────────────────────┘
//
//   读冲突发生频率：每个 dotIdx 迭代 1 次，共 K/BK × BK = K 次
//   写冲突发生频率：每次 tile 加载 1 次，共 K/BK 次
//   → 读冲突次数是写冲突的 BK=8 倍，转置消除读冲突的收益远大于引入写冲突的代价
//
// ── reinterpret_cast vs static_cast ─────────────────────────────────────────
//
//   static_cast：编译期类型检查，只允许语义上合法的转换
//     static_cast<float>(intVal)      // ✓ int→float，有值转换
//     static_cast<float4*>(floatPtr)  // ✗ 编译报错，不相关指针类型
//
//   reinterpret_cast：不做任何转换，只重新解释内存的比特模式
//     reinterpret_cast<float4*>(floatPtr)  // ✓ 把 float* 当 float4* 读
//     没有类型检查，没有值转换，责任完全在程序员
//     地址未对齐或类型不兼容 → 未定义行为，编译器不报错
//
//   本 kernel 必须用 reinterpret_cast：float* 和 float4* 是不相关指针类型，
//   static_cast 无法完成此转换
//
// ── 与 kernel_5 相比的三处改进 ──────────────────────────────────────────────
//
//   ① GMEM→SMEM 加载改用 float4（执行效率优化，非带宽利用率优化）
//       kernel_5：标量 load + loadOffset 循环，每 4 个 float 需 4 条指令 + 循环控制
//       kernel_6：float4 一条指令取 4 个 float，消除循环
//       sector 利用率两者相同（均 100%），收益在于减少指令发射压力
//
//   ② As 存储时转置（行主序 → 列主序）← 主要收益
//       kernel_5：As[row*BK+col]，计算阶段按列读 As → SMEM bank conflict
//       kernel_6：As[col*BM+row]，计算阶段连续读 As → 消除 bank conflict，
//                 FMA 单元不再空等 SMEM → 执行效率大幅提升
//
//   ③ 写回 C 也用 float4
//       kernel_5：逐元素 32-bit store
//       kernel_6：每次写 4 个元素，128-bit store，store 指令数减少到 1/4
// ══════════════════════════════════════════════════════════════════════════════

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
gemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    // 表达式拆解：
    //   &A[innerRowA * K + innerColA * 4]          → float*   取 A 中某元素的地址
    //   reinterpret_cast<float4*>(float*)           → float4*  重解释为 float4 指针
    //   (float4*)[0]  等价于 *(ptr + 0)            → float4   解引用，取结构体值
    //   [0] 是指针下标（解引用），不是取地址；ptr[0] = *(ptr+0)，不是 &(ptr+0)
    //   解引用得到的类型由指针类型决定：float* 解引用→float，float4* 解引用→float4
    //   reinterpret_cast 把内存强行解释为 float4*，解引用因此返回完整的 float4 结构体
    //   reinterpret_cast 只重解释地址，不移动数据；从该地址起连续 16 字节填入 x/y/z/w：
    //     地址: p      p+4    p+8    p+12
    //     内存: [f0]   [f1]   [f2]   [f3]
    //            ↓      ↓      ↓      ↓
    //           tmp.x  tmp.y  tmp.z  tmp.w
    //   A 是行主序一维数组，&A[...] 之后 4 个 float 天然连续，前提成立
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    // 读取：tmp.x/y/z/w = A[innerRowA][innerColA*4 + 0/1/2/3]
    //         即 A 中坐标 (row=innerRowA, col=innerColA*4+k)
    // 写入：As[(col)*BM + row]，即列主序 As[col][row]
    //   A[row][col]  →  As[col][row]   行列互换 = 转置
    // 转置目的：计算阶段按 dotIdx 读 As 时变为连续地址，消除 SMEM bank conflict
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    // 显式寄存器缓存（regM/regN）的 SMEM 读次数分析：
    //   外层 dotIdx 循环（BK=16次），每轮：
    //     加载 regM[TM=8]：8 次 SMEM 读（As），每个 regM[i] 被 TN=8 次内层循环复用 → 仅寄存器访问
    //     加载 regN[TN=8]：8 次 SMEM 读（Bs），每个 regN[i] 被 TM=8 次 resIdxM 循环复用 → 仅寄存器访问
    //   总 SMEM 读：As = BK×TM = 128 次；Bs = BK×TN = 128 次；每个元素恰好读一次
    //   对比 v2 的 tmpAs/tmpBs：Bs 被读 TM×TN×BK = 1024 次（多 8 倍，见 v2 计算部分注释）
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
gemmVectorize_v2(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {

  // 该block的起始行和列，一个Block负责BM*BN个元素的乘积累加
  const uint initRow {blockIdx.y * BM};
  const uint initCol {blockIdx.x * BN};

  // 分配该block需要的静态SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 以下 assert 保证 block 内部参数自洽（tile 内部无边界问题）：
  //   一旦数据从 GMEM 加载到 SMEM，无法再处理边界情况（SMEM 大小固定为 BM×BK/BK×BN）；
  //   外层 M/N/K 不整除 BM/BN/BK 的边界，必须在 GMEM→SMEM 加载阶段用条件判断处理。
  //   当前 kernel 假设 M%BM==0, N%BN==0, K%BK==0，未做外层边界处理。
  // BK 必须是 4 的倍数，才能用 float4（每次加载 4 个 float）整除 As 的列
  static_assert(BK % 4 == 0 && "As的列BK不是float4向量的倍数");
  // blockDim.x 必须整除 BK/4（每行需要的float4列组数），保证每行 As 被线程完整覆盖，无剩余列
  assert(blockDim.x % (BK / 4) == 0 && "As 的行不能被一个block中的线程完整覆盖，有剩余列，列方向不能被整数");
  // BN 必须是 4 的倍数，才能用 float4 整除 Bs 的列
  static_assert(BN % 4 == 0 && "Bs的列BN不是float4向量的倍数");
  // blockDim.x 必须整除 BN/4，保证每行 Bs 被线程完整覆盖，无剩余列
  assert(blockDim.x % (BN / 4) == 0 && "Bs 的行不能被一个block中的线程完整覆盖，有剩余列，列方向不能被整除");
  static_assert(BM == BN &&  "Bs维度和As一样");
  assert(BM * BK % (blockDim.x * 4) == 0 && "As是和Bs的列不能被一个block中的线程在多个轮次后完整覆盖，有剩余行，行方向不能被整除");

  //写入AsBs的每一行需要多少线程
  const uint threadNumqryPerRowAs {BK / 4};
  const uint threadNumqryPerRowBs {BN / 4};
  // 一个block能写入多少行
  // 这种写法是循环AS、BS的行，每一个轮次跳过rowNumPerBlock行，
  // 也可以直接计算要循环几次:
  //  assert(BM * BK % (blockDim.x * 4) == 0);
  //  const uint recycleNum {BM * BK / (blockDim.x * 4)};
  // 展开验证：
  // rowNumPerBlockAs = blockDim.x / (BK/4) = blockDim.x * 4 / BK
  // rowIdx 循环次数 = BM / rowNumPerBlockAs
  //                = BM / (blockDim.x * 4 / BK)
  //                = BM * BK / (blockDim.x * 4)   ← 就是 recycleNum
  const uint rowNumPerBlockAs {blockDim.x / threadNumqryPerRowAs};
  const uint rowNumPerBlockBs {blockDim.x / threadNumqryPerRowBs};
  // 在一个轮次中该线程负责写入的列组和行
  // 每个线程处理float4列向量
  const uint innerColGroupAs {threadIdx.x % threadNumqryPerRowAs};
  const uint innerRowAs {threadIdx.x / threadNumqryPerRowAs};
  const uint innerColGroupBs {threadIdx.x % threadNumqryPerRowBs};
  const uint innerRowBs {threadIdx.x / threadNumqryPerRowBs};

  // 该线程负责的TM * TN的元素的计算
  static_assert(BN % TN == 0 &&  "列组不是整数");
  // blockDim.x 必须整除 BN/TN（计算阶段每行的列组数），保证所有列组被线程完整覆盖，无剩余列
  assert(blockDim.x % (BN / TN) == 0 &&  "(BM,BN)的行不能被一个block中的线程完整覆盖，无剩余列，列方向不能被整数");
  assert(BM * BN %  == (blockDim.x * TM * TN) &&  "(BM,BN)的列不能被一个block中的线程在一个轮次后完整覆盖，有剩余行，列方向不能被整除");
  const uint threadColGroop {threadIdx.x % (BN / TN)};
  const uint threadRowGroup {threadIdx.x / (BN / TN)};
  // threadResult 必须在外层 K 循环之外声明并初始化：
  //   GEMM 结果 = 沿 K 方向所有 tile 的累加，每轮 outterIdx 都往同一个 threadResult 中 +=
  //   若放在 K 循环内，每轮 tile 都重置为 0，只保留最后一个 tile 的结果 → 计算错误
  float threadResult[TM * TN] {0.0f};
  float tmpAs[TM];
  float tmpBs[TN];

  // ── 加载阶段线程分配：跨步（当前）vs 连续（块状）对比 ──────────────────────────
  //
  //   跨步方案（当前）：相邻线程负责相邻 SMEM 位置，每线程自身跨步迭代
  //     thread 0 → SMEM[0..3]，thread 1 → SMEM[4..7]，thread 2 → SMEM[8..11]，...
  //
  //   连续方案（块状）：每线程负责连续一整块 SMEM，相邻线程间距 = chunk 大小
  //     thread 0 → SMEM[0..chunk-1]，thread 1 → SMEM[chunk..2*chunk-1]，...
  //
  // ── GMEM 读（A/B → 寄存器）──────────────────────────────────────────────────
  //
  //   跨步：相邻线程读 GMEM 相邻地址（同一行内连续 float4）→ 合并为少量事务 ✓
  //     float4 进一步放大收益：每次事务传输 16 字节，事务数降为标量的 1/4
  //
  //   连续：每线程负责一块连续 SMEM，对应 GMEM 中跨行的元素（行间距 = K/N）
  //     相邻线程 GMEM 地址不连续 → 无法合并 → 事务数暴增（最多 blockDim.x 倍）✗
  //
  // ── SMEM 写（GMEM → As/Bs，经寄存器中转）───────────────────────────────────
  //
  //   跨步：相邻线程写相邻 SMEM 地址，线程间步长 = 4（float4）
  //     .x 元素 bank 步长 = 4，周期 = 32/4 = 8 → 4-way conflict
  //
  //   连续（chunk=8）：线程间步长 = 8
  //     .x 元素 bank 步长 = 8，周期 = 32/8 = 4 → 8-way conflict（更差）✗
  //
  //   连续（chunk=BN=128，每线程负责完整一行）：线程间步长 = 128
  //     128 % 32 = 0 → 所有线程撞同一 bank → 32-way conflict（灾难性）✗✗
  //
  //   结论：步长越小，bank conflict 越少；跨步的步长（4）< 连续的步长（≥8）→ 跨步更优
  //
  // ── 综合对比 ────────────────────────────────────────────────────────────────
  //
  //   ┌──────────┬──────────────────────────┬──────────────────────────────┐
  //   │          │ GMEM 读（A/B）           │ SMEM 写（As/Bs）             │
  //   ├──────────┼──────────────────────────┼──────────────────────────────┤
  //   │ 跨步     │ 合并 ✓（少量事务）       │ 4-way conflict（步长=4）     │
  //   │ 连续     │ 不合并 ✗（事务数×N）    │ ≥8-way conflict（步长≥8）✗  │
  //   └──────────┴──────────────────────────┴──────────────────────────────┘
  //
  //   跨步在两个维度上均优于连续：GMEM 合并收益（~1000×）>> SMEM conflict 代价（~8×）
  //   加载阶段选跨步分配，与计算阶段选块状分配（便于寄存器复用），各有其适用场景
  //
  // 沿K方向的循环
  for (uint outterIdx{}; outterIdx < K; outterIdx+=BK) {
    // 写入As、Bs的一个轮次
    for (uint rowIdx{}; rowIdx < BM; rowIdx+=rowNumPerBlockAs) {
      // initRow在一个block中不变、colIdx在一次循环中不变
      // BK=8，1个warp32个线程的innerColGroupAs列组是0，1，...，0，1，列是0，4，...，0，4，innerRowAs行是0，0，1，1，...，15，15，
      // 读取A的时候：
      //  一个线程取4个float32，就是16字节，2个线程就是32个字节，并且同一行。线程对使用一个sector。
      //  相邻的线程对相差K个float32，4K个字节，当K大于32时，就不能共用一个cache line，无法合并访问事务，32个线程16个线程对一共16个访问事务
      // A写入As的时候：
      //   As[(innerRowAs + rowIdx) * BK + innerColGroupAs * 4] = colVecAs.x;
      //   线程0写入（0，0），线程1写入（0，4），线程2写入（1,0），线程3写入（1,4）
      //   那么，间隔4个元素，，每 8 个线程 bank 重复，所以4-way bank conflict
      // BM=128,，1个warp32个线程的innerColGroupAs列组是0，1，...，0，1，列是0，4*128，...，0，4*128，innerRowAs行是0，0，1，1，...，15，15，
      // A的转置写入As的时候：
      //   线程0写入（0，0），线程1写入（4*128，0），，线程2写入（0,1），线程3写入（4*128,1）,写了2行，2-way bank conflict
      float4 colVecAs;
      if ((innerColGroupAs * 4 + outterIdx) + 3 < K  && (initRow + innerRowAs + rowIdx) < M) {
        colVecAs = reinterpret_cast<float4 *>(&A[(initRow + innerRowAs + rowIdx) * K + innerColGroupAs * 4 + outterIdx])[0];
      } else if ((innerColGroupAs * 4 + outterIdx) + 2 < K  && (initRow + innerRowAs + rowIdx) < M) {
        colVecAs.x = A[(initRow + innerRowAs + rowIdx) * K + innerColGroupAs * 4 + outterIdx];
        colVecAs.y = A[(initRow + innerRowAs + rowIdx) * K + innerColGroupAs * 4 + outterIdx + 1];
        colVecAs.z = A[(initRow + innerRowAs + rowIdx) * K + innerColGroupAs * 4 + outterIdx + 2];
        colVecAs.w = 0.0f;
      } else if ((innerColGroupAs * 4 + outterIdx) + 1 < K  && (initRow + innerRowAs + rowIdx) < M) {
        colVecAs.x = A[(initRow + innerRowAs + rowIdx) * K + innerColGroupAs * 4 + outterIdx];
        colVecAs.y = A[(initRow + innerRowAs + rowIdx) * K + innerColGroupAs * 4 + outterIdx + 1];
        colVecAs.z = 0.0f;
        colVecAs.w = 0.0f;
      } else if ((innerColGroupAs * 4 + outterIdx)  < K  && (initRow + innerRowAs + rowIdx) < M) {
        colVecAs.x = A[(initRow + innerRowAs + rowIdx) * K + innerColGroupAs * 4 + outterIdx];
        colVecAs.y = 0.0f;
        colVecAs.z = 0.0f;
        colVecAs.w = 0.0f;
      } else {
        colVecAs.x = 0.0f;
        colVecAs.y = 0.0f;
        colVecAs.z = 0.0f;
        colVecAs.w = 0.0f;
      }
      //将A写入As，行主序
      // As[(innerRowAs + rowIdx) * BK + innerColGroupAs * 4] = colVecAs.x;
      // As[(innerRowAs + rowIdx) * BK + innerColGroupAs * 4 + 1] = colVecAs.y;
      // As[(innerRowAs + rowIdx) * BK + innerColGroupAs * 4 + 2] = colVecAs.z;
      // As[(innerRowAs + rowIdx) * BK + innerColGroupAs * 4 + 3] = colVecAs.w;

      //将A的转置写入As，列主序
      As[(innerColGroupAs * 4) * BM + (innerRowAs + rowIdx)] = colVecAs.x;
      As[(innerColGroupAs * 4 + 1) * BM + (innerRowAs + rowIdx)] = colVecAs.y;
      As[(innerColGroupAs * 4 + 2) * BM + (innerRowAs + rowIdx)] = colVecAs.z;
      As[(innerColGroupAs * 4+ 3) * BM + (innerRowAs + rowIdx)] = colVecAs.w;
    }

    for (uint rowIdx{}; rowIdx < BK; rowIdx+=rowNumPerBlockBs) {
      // initCol在一个block中不变、colIdx在一次循环中不变
      // BN=128，1个warp32个线程的innerColGroupBs列组是0，1，2，32，...，0，1，2，32，列是0，4，8，128，...，0，4，8，128，innerRowBs行是0，0，0，0，...，0，0(同一行)，
      // 读取B的时候：
      //   一个线程取4个float32，就是16字节，2个线程就是32个字节，并且同一行。线程对使用一个sector。
      //   所有的32个线程在同一行，元素连续，所以可以合并访问事务，8个线程共用一个cache line，所以32个线程一共4个访问事务
      // 写入Bs的时候：
      //   线程0写（0，0），线程1写（0，4），...
      //   那么，间隔4个元素，，每 8 个线程 bank 重复，所以4-way bank conflict
      float4 colVecBs;
      if ((innerRowBs + outterIdx + rowIdx) < K  && (innerColGroupBs * 4 + initCol) + 3 < N) {
        colVecBs = reinterpret_cast<float4 *>(&B[(innerRowBs + outterIdx + rowIdx) * N + innerColGroupBs * 4 + initCol])[0];
      } else if ((innerRowBs + outterIdx + rowIdx) < K  && (innerColGroupBs * 4 + initCol) + 2 < N) {
        colVecBs.x = B[(innerRowBs + outterIdx + rowIdx) * N + innerColGroupBs * 4 + initCol];
        colVecBs.y = B[(innerRowBs + outterIdx + rowIdx) * N + innerColGroupBs * 4 + initCol + 1];
        colVecBs.z = B[(innerRowBs + outterIdx + rowIdx) * N + innerColGroupBs * 4 + initCol + 2];
        colVecBs.w = 0.0f;
      } else if ((innerRowBs + outterIdx + rowIdx) < K  && (innerColGroupBs * 4 + initCol) + 1 < N) {
        colVecBs.x = B[(innerRowBs + outterIdx + rowIdx) * N + innerColGroupBs * 4 + initCol];
        colVecBs.y = B[(innerRowBs + outterIdx + rowIdx) * N + innerColGroupBs * 4 + initCol + 1];
        colVecBs.z = 0.0f;
        colVecBs.w = 0.0f;
      } else if ((innerRowBs + outterIdx + rowIdx)  < K  && (innerColGroupBs * 4 + initCol)  < N) {
        colVecBs.x = B[(innerRowBs + outterIdx + rowIdx) * N + innerColGroupBs * 4 + initCol];
        colVecBs.y = 0.0f;
        colVecBs.z = 0.0f;
        colVecBs.w = 0.0f;
      } else {
        colVecBs.x = 0.0f;
        colVecBs.y = 0.0f;
        colVecBs.z = 0.0f;
        colVecBs.w = 0.0f;
      }
      //写入Bs
      Bs[(innerRowBs + rowIdx) * BN + innerColGroupBs * 4] = colVecBs.x;
      Bs[(innerRowBs + rowIdx) * BN + innerColGroupBs * 4 + 1] = colVecBs.y;
      Bs[(innerRowBs + rowIdx) * BN + innerColGroupBs * 4 + 2] = colVecBs.z;
      Bs[(innerRowBs + rowIdx) * BN + innerColGroupBs * 4 + 3] = colVecBs.w;
    }

    // 同步
    __syncthreads();


    // // tmpBs/tmpAs 的 SMEM 读次数分析（对比 v1 的 regM/regN）：
    // //   tmpBs[BK] 在 colIdx 循环内加载，但 Bs 索引不含 rowIdx
    // //   → 同一份 tmpBs 在每个 rowIdx 迭代中都重新从 SMEM 加载，共加载 TM=8 次
    // //   总 Bs SMEM 读：TM×TN×BK = 8×8×16 = 1024 次（比 v1 多 8 倍）
    // //   v1 的 regN 外层是 dotIdx 循环，regN[i] 在 TM 次 resIdxM 循环中复用 → Bs 只读 BK×TN=128 次
    // //   根本原因：v2 循环顺序是 rowIdx→colIdx→innerIdx，Bs 不依赖 rowIdx 却被困在其内部
    // //             v1 循环顺序是 dotIdx→(regM+regN)→resIdxM→resIdxN，结构上保证每元素只读一次
    // // 对写入的As、Bs求乘积累加
    // for (uint rowIdx{}; rowIdx < TM; ++rowIdx) {
    //   // tmpAs 必须在 colIdx 循环之外声明并加载：
    //   //   tmpAs 只依赖 rowIdx（As 的行），与 colIdx 无关；
    //   //   若放在 colIdx 内，每轮 colIdx 都重新从 SMEM 加载 BK 次，共多读 (TN-1)*BK 次 → 浪费
    //   //   放在 colIdx 外，每个 rowIdx 只加载一次，TN 个 colIdx 共用同一份 tmpAs ✓
    //   float tmpAs[BK];
    //   for (uint colIdxAs{}; colIdxAs < BK; ++colIdxAs) {
    //     tmpAs[colIdxAs] = As[colIdxAs * BM + (threadRowGroup * TM + rowIdx)];
    //   }
    //   float tmpBs[BK];
    //   for (uint colIdx{}; colIdx < TN; ++colIdx) {
    //     for (uint rowIdxBs{}; rowIdxBs < BK; ++rowIdxBs) {
    //       tmpBs[rowIdxBs] = Bs[rowIdxBs * BN + threadColGroop  * TN + colIdx];
    //     }
    //
    //     for (uint innerIdx{}; innerIdx < BK; ++innerIdx) {
    //       threadResult[rowIdx * TN + colIdx] += tmpAs[innerIdx] * tmpBs[innerIdx];
    //     }
    //   }
    // }


    // ── 计算阶段线程分配：块状（当前）vs 跨步（loading style）对比 ──────────────
    //
    //   块状（当前）：每个线程负责连续的 TM×TN 子块
    //     thread 0 → C[0..7][0..7]，thread 1 → C[0..7][8..15]，...
    //     tmpAs[TM] 每个 dotIdx 加载一次，被 TN 次内层循环复用（寄存器访问）
    //     tmpBs[TN] 每个 dotIdx 加载一次，被 TM 次外层循环复用（寄存器访问）
    //     每 dotIdx 的 SMEM 读 = TM+TN = 16 次，完成 TM×TN = 64 次 MAD
    //     总 SMEM 读/thread = BK×(TM+TN) = 16×16 = 256 次
    //
    //   跨步（loading style）：相邻线程负责相邻列，每线程的元素分散在不同行
    //     thread 0 → C[0][0], C[0][32], ...，thread 1 → C[0][1], C[0][33], ...
    //     负责的元素分散 → 无法将 As/Bs 缓存在寄存器里跨元素复用
    //     每个输出元素都需重新读 BK 次 SMEM
    //     总 SMEM 读/thread = TM×TN×BK = 8×8×16 = 1024 次（块状的 4 倍）
    //
    //   跨步唯一优势：C 写回时相邻线程写相邻列 → 完全合并（32线程覆盖连续32列）
    //   块状的 C 写回：相邻线程列间距 = TN=8 → 部分跨步
    //   但 C 写回每 tile 只发生一次，SMEM 读每 tile 发生 BK 次 → 写回收益远小于 SMEM 读浪费
    //
    //   结论：计算阶段用块状分配，瓶颈在 SMEM 访问次数而非 C 写回合并
    //         加载阶段用跨步分配，目的是 GMEM 合并读，场景不同

    // 对写入的As、Bs求乘积累加
    for (uint colIdx{}; colIdx < BK; ++colIdx) {
      for (uint rowNumIdx{}; rowNumIdx < TM; ++rowNumIdx) {
        // 读取AS
        // 一个轮次，colIdx、rowNumIdx不变，threadRowGroup变化
        // 一个warp32个线程，BN / TN= 128 / 8 =16，threadColGroop列组是0，1，2，3，...，15，threadRowGroup行组是0，0，0，0...，1，1，1，1
        // 前16线程访问1个元素，后16个线程访问另外一个元素，相差8个元素，无 bank conflict
        // ---- 列主序 vs 行主序的 bank conflict 对比 ----
        // 列主序（当前）：As[colIdx*BM + threadRowGroup*TM + rowNumIdx]
        //   threadColGroup 不出现 → 同 threadRowGroup 组内所有线程访问同一地址 → broadcast，无 conflict
        //   threadRowGroup 0 vs 1 地址相差 TM=8 个元素，bank 相差 8 → 不同 bank，无 conflict
        // 行主序（对比）：As[(threadRowGroup*TM+rowNumIdx)*BK + colIdx]
        //   threadRowGroup 0 vs 1 地址相差 TM*BK=8*16=128=4×32 个元素 → bank 相同但地址不同 → 2-way conflict
        //   列主序（转置存储）正是为了消除此 conflict
        tmpAs[rowNumIdx] = As[colIdx * BM + (threadRowGroup * TM + rowNumIdx)];
      }

      for (uint colNumIdx{}; colNumIdx < TN; ++colNumIdx) {
        // 读取BS
        // 一个轮次，colIdx、colNumIdx不变，threadColGroop变化
        // 一个warp32个线程，BN / TN= 128 / 8 =16，threadColGroop列组是0，1，2，3，...，15，0，1，2，3，...，15，threadRowGroup行组是0，0，0，0...，1，1，1，1
        // 每个线程相差8个元素，一行访问4个bank，2个线程访问同一个元素，只需要看前16个线程， 16 / 4 = 4-way conflict
        tmpBs[colNumIdx] = Bs[colIdx * BN + threadColGroop  * TN + colNumIdx];
      }

      for (uint rowNumIdx{}; rowNumIdx < TM; ++rowNumIdx) {
        for (uint colNumIdx{}; colNumIdx < TN; ++colNumIdx) {
          threadResult[rowNumIdx * TN + colNumIdx] += tmpAs[rowNumIdx] * tmpBs[colNumIdx];
        }
      }
    }

    // 同步
    __syncthreads();
  }

  // //  写回C,非向量加载
  // for (uint rowIdx{}; rowIdx < TM; ++rowIdx) {
  //   for (uint colIdx{}; colIdx < TN; ++colIdx) {
  //     if ((initRow + threadRowGroup * TM + rowIdx) < M && initCol + threadColGroop * TN + colIdx < N) {
  //       C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx] =  alpha * threadResult[rowIdx * TN + colIdx] + beta * C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx];
  //     }
  //   }
  // }

  // 写回C，向量加载
  // C[index] 不是数组退化：kernel 参数声明为 float *C，本身已是指针
  //   C[index] 等价于 *(C + index)，是指针下标访问，无退化发生
  //   数组退化（array decay）仅发生在 float arr[N] 类型的数组在表达式中被当作指针使用时
  assert(TN % 4 == 0 && "该线程中tn个结果不能被4整除");
  for (uint rowIdx{}; rowIdx < TM; ++rowIdx) {
    for (uint colIdx{}; colIdx < TN; colIdx+=4) {
      if ((initRow + threadRowGroup * TM + rowIdx) < M && initCol + threadColGroop * TN + colIdx + 3 < N) {
        float4 cvec4 { reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx])[0] };
        cvec4.x =  alpha * threadResult[rowIdx * TN + colIdx] + beta * cvec4.x;
        cvec4.y =  alpha * threadResult[rowIdx * TN + colIdx + 1] + beta * cvec4.y;
        cvec4.z =  alpha * threadResult[rowIdx * TN + colIdx + 2] + beta * cvec4.z;
        cvec4.w =  alpha * threadResult[rowIdx * TN + colIdx + 3] + beta * cvec4.w;
        reinterpret_cast<float4 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx])[0] = cvec4;
      } else if ((initRow + threadRowGroup * TM + rowIdx) < M && initCol + threadColGroop * TN + colIdx + 2 < N) {
        // float3：PTX 无 st.global.v3.f32 指令（只有 v1/v2/v4）
        // 编译器拆成 st.global.v2.f32（x,y）+ st.global.f32（z），共 2 条指令
        // 功能正确，边界情况极少，性能影响可忽略
        float3 cvec3 { reinterpret_cast<float3 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx])[0] };
        cvec3.x =  alpha * threadResult[rowIdx * TN + colIdx] + beta * cvec3.x;
        cvec3.y =  alpha * threadResult[rowIdx * TN + colIdx + 1] + beta * cvec3.y;
        cvec3.z =  alpha * threadResult[rowIdx * TN + colIdx + 2] + beta * cvec3.z;
        reinterpret_cast<float3 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx])[0] = cvec3;
      } else if ((initRow + threadRowGroup * TM + rowIdx) < M && initCol + threadColGroop * TN + colIdx + 1 < N) {
        float2 cvec2 { reinterpret_cast<float2 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx])[0] };
        cvec2.x =  alpha * threadResult[rowIdx * TN + colIdx] + beta * cvec2.x;
        cvec2.y =  alpha * threadResult[rowIdx * TN + colIdx + 1] + beta * cvec2.y;
        reinterpret_cast<float2 *>(&C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx])[0] = cvec2;
      } else if ((initRow + threadRowGroup * TM + rowIdx) < M && initCol + threadColGroop * TN + colIdx  < N) {
        C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx] =  alpha * threadResult[rowIdx * TN + colIdx] + beta * C[(initRow + threadRowGroup * TM + rowIdx) * N + initCol + threadColGroop * TN + colIdx];
      }
    }
  }
}

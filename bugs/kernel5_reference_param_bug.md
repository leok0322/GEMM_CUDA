# Bug: `__global__` kernel 引用参数导致 illegal memory access

## 现象

```
[CUDA ERROR] at file /home/liam/cpp_linux/GEMM_CUDA/gemm.cu:163:
an illegal memory access was encountered
```

测试用例：`m=n=k=128, alpha=0.5, beta=3`，kernel 5 第一个尺寸即崩溃。

---

## 根本原因

kernel 签名将 `offsetCol` / `offsetRow` 声明为引用参数：

```cpp
// ❌ 错误写法
gemm2DBlocktiling_v3(..., const uint& offsetCol, const uint& offsetRow)
```

引用的底层是指针，指向的是**主机栈**上的变量地址。  
CUDA runtime 将 kernel 参数按值拷贝到设备常量内存（constant memory）再分发给线程，拷贝的是引用所持有的**地址值**本身（一个主机内存地址）。  
GPU 线程在执行 `(blockIdx.y + offsetRow) * BM` 时，用这个地址解引用 → 访问主机内存 → **illegal memory access**。

```cpp
// ✓ 正确写法：值传参，runtime 直接拷贝数值
gemm2DBlocktiling_v3(..., const uint offsetCol, const uint offsetRow)
```

> **结论：`__global__` kernel 参数只能是值类型或设备指针，不能是引用。**

---

## Debug 过程

**第一步：定位错误类型**

报错位置是 `cudaDeviceSynchronize()`，捕获的是**异步错误**——错误发生在 kernel 内部，不是 launch 时参数非法。  
`illegal memory access` = GPU 线程访问了无权访问的内存地址。

**第二步：排除边界越界**

测试尺寸 `m=n=k=128`，恰好等于 `BM=BN=128`，矩阵整除，不存在边界块，`BOUNDARY=false` 路径也不会有下标越界。  
→ 排除数组下标越界。

**第三步：缩小到参数传递**

`illegal memory access` 常见原因：
- 数组下标越界 → 已排除
- 访问主机内存地址 → 可疑
- 空指针解引用 → A/B/C 均为正常设备指针

怀疑指向**参数本身**，读 kernel 签名，发现：

```cpp
const uint& offsetCol, const uint& offsetRow   // ← 引用！
```

**第四步：确认根因**

引用 = 主机栈变量的地址，GPU 线程解引用时访问主机内存 → 确认是根因。

**关键线索链：**

```
cudaDeviceSynchronize() 报错（async，kernel 内部）
→ m=128 无越界风险
→ 怀疑参数传递
→ 发现 const uint& → 主机地址传入 GPU
→ 确认：illegal memory access
```

---

## 修复

```cpp
// 将引用改为值传参
gemm2DBlocktiling_v3(int M, int N, int K, float alpha, const float *A,
                 const float *B, float beta, float *C,
                 const uint offsetCol, const uint offsetRow)
```

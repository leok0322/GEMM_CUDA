#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
// runner.cuh 中声明了 run_kernel 的函数签名
// 编译 gemm.cu 时只需要看到声明，编译器在调用处暂时填 0（占位），
// 并在 gemm.cu.o 的重定位表中记录："此处需要填 run_kernel 的真实地址"
// 链接阶段 ld 扫描所有 .o 的符号表，在 runner.cu.o 中找到 run_kernel 的实现，
// 将其真实地址回填到 gemm.cu.o 中所有调用处，完成重定位（relocation）
// 这依赖 CMakeLists.txt 中 add_executable(gemm gemm.cu ${SRC}) 把 runner.cu 也纳入同一 target
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))


const std::string errLogFile = "matrixValidationFailure.txt";

// argc: 命令行参数个数（含程序名本身）
// argv: 参数字符串数组
//   argv[0]: 程序路径，由操作系统自动填入（如 ./build/gemm），不需要用户传递
//   argv[1]: kernel 编号，由用户传入（如 ./build/gemm 3）
// argc != 2 即用户未传入 kernel 编号
int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // 通过环境变量 DEVICE 指定使用哪块 GPU，默认使用 GPU 0
  // getenv("DEVICE")：读取环境变量，未设置时返回 NULL
  // atoi()：将字符串转为 int（如 "1" -> 1），处理的是 char* 字符串
  //   不等价于 static_cast<int>()，后者只做数值类型间转换（如 float->int），无法解析字符串
  //   更安全的 C++ 替代：std::stoi()，输入非法时抛出异常而非返回 0
  // 用法：DEVICE=1 ./build/gemm 3   指定使用 GPU 1 运行 kernel 3
  //   KEY=VALUE ./程序 的 shell 语法：环境变量仅对本次进程生效，不污染当前 shell
  //   等价写法：export DEVICE=1 && ./build/gemm 3（export 会持久影响当前 shell 全局）
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = std::atoi(getenv("DEVICE"));
  }
  // cudaSetDevice：切换当前线程操作的 GPU，多卡机器上必须在任何 CUDA 调用之前设置
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  // SIZE.size()-1 是最后一个元素的下标，即取 SIZE 中最大的尺寸（4096）
  // 目的：按最大尺寸一次性分配内存，后续循环测试不同 size 时复用，避免反复 malloc/free
  // 等价写法：SIZE.back()（更惯用的 C++ 写法）
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  // malloc 返回 void*（通用指针，表示"一块未知类型的内存"）
  // 此处需要的转换方向是 void* → float*：
  //   C++ 中：void* → float* 不允许隐式转换，必须显式强转 (float*)
  //   C   中：void* → float* 允许隐式转换，malloc 返回值可直接赋给 float*，无需强转
  // 反方向 float* → void* 两种语言都允许隐式转换，无需强转
  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50;
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};
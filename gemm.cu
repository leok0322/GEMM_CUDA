#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
// runner.cuh 中声明了 run_kernel 的函数签名
// 编译 gemm.cu 时只需要看到声明，编译器在调用处暂时填 0（占位），
// 并在 gemm.cu.o 的重定位表中记录："此处需要填 run_kernel 的真实地址"
// 链接阶段 ld 扫描所有 .o 的符号表，在 runner.cu.o 中找到 run_kernel 的实现，
// 将其真实地址回填到 gemm.cu.o 中所有调用处，完成重定位（relocation）
// 这依赖 CMakeLists.txt 中 add_executable(gemm gemm.cu ${SRC}) 把 runner.cu 也纳入同一 target
//
// 使用尖括号 <runner.cuh> 而非双引号 "runner.cuh"：
//   双引号搜索顺序：① 当前源文件所在目录 → ② -I 路径 → ③ 系统默认路径（/usr/include 等）
//   尖括号搜索顺序：① -I 路径 → ② 系统默认路径（/usr/include 等），不搜索当前目录
//   runner.cuh 在 src/ 下，已通过 target_include_directories(gemm PRIVATE src/) 加入 -I
//   两种写法在此处效果相同，尖括号是惯例：表示"通过构建系统管理的路径"而非"相对路径的本地文件"
#include <iomanip>
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

  // cublasHandle_t：cuBLAS 库的上下文句柄，本质是一个不透明指针（opaque pointer）
  //   typedef struct cublasContext* cublasHandle_t;
  //   内部 cublasContext 结构体由 cuBLAS 库管理，用户不直接访问，包含：
  //     - 当前绑定的 CUDA stream（默认 stream 0）
  //     - 当前 GPU 设备信息
  //     - 工作区内存（workspace）指针
  //     - 数学模式（math mode）：CUBLAS_DEFAULT_MATH / CUBLAS_TENSOR_OP_MATH 等
  //     - 原子操作模式、日志设置等
  //   所有 cuBLAS API 调用都需要传入此 handle，类似 OpenGL 的上下文
  cublasHandle_t handle;

  // cublasCreate 函数签名：cublasStatus_t cublasCreate(cublasHandle_t *handle)
  //   - 参数：cublasHandle_t* 指针，函数内部分配 cublasContext 并将地址写入 handle
  //   - 返回：cublasStatus_t 枚举，CUBLAS_STATUS_SUCCESS=0 表示成功，非 0 表示失败
  //   - 内部操作：初始化 cuBLAS 库、分配 GPU 资源、绑定当前 CUDA 设备
  //   - if(cublasCreate(&handle))：返回非 0（失败）时进入错误处理，0（成功）不进入
  //   - 对应销毁：程序结束前需调用 cublasDestroy(handle) 释放资源
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
      try {
        run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                   handle); // cuBLAS
        // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
        cudaCheck(cudaGetLastError());
        run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                   handle); // Executes the kernel, modifies the result matrix
        // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
        cudaCheck(cudaGetLastError());
        // 再等待 GPU 执行完毕，捕获 kernel 运行时的异步错误
        // 若顺序反过来，cudaDeviceSynchronize 会消费掉异步错误，cudaGetLastError 可能漏报
        cudaCheck(cudaDeviceSynchronize());
      } catch (const std::exception &e) {
        // run_kernel 的 default 分支会 throw std::invalid_argument
        // 未捕获时触发 std::terminate()，输出系统错误信息，不友好
        // 捕获后打印可读错误信息并干净退出，更优雅
        //
        // catch(const std::exception &e)：
        //   std::exception 是所有标准异常的基类，捕获基类引用可接住所有标准异常
        //   const &：引用避免拷贝，const 保证不修改异常对象
        //   e.what()：虚函数，多态调用到具体子类实现，返回异常描述字符串
        printf("%s\n", e.what());
        exit(EXIT_FAILURE);
      }
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        // 矩阵太大时打印到终端不现实，只对小矩阵（m<=128）写入文件便于调试
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          // std::ofstream：输出文件流，用于写文件
          // fs.open(errLogFile)：打开 errLogFile（"matrixValidationFailure.txt"）
          //   默认以覆盖模式（trunc）打开，每次验证失败都会覆盖上次内容
          std::ofstream fs;
          fs.open(errLogFile);
          // << 运算符：将字符串/矩阵内容写入文件，与 std::cout 用法相同
          // 依次写入 A、B、kernel 输出 C、cuBLAS 参考结果 C_ref
          // 便于对比：看 A、B 的输入是什么，kernel 算出什么，正确结果应该是什么
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
          // 【RAII：资源生命周期与对象生命周期绑定】
          // std::ofstream 是 RAII 类，离开 {} 作用域时析构函数自动调用 close()
          // 刷新缓冲区（flush）并释放文件句柄，无需手动 fs.close()
          //
          // 适用范围：栈上对象（ofstream / ifstream / unique_ptr / lock_guard 等）
          //   离开 {} 一定析构 → 资源一定释放
          // 不适用：new / malloc 分配的堆内存，需手动 delete / free
        }
        // EXIT_FAILURE：标准宏，值为 1，表示程序异常退出
        // 验证失败直接终止，不继续跑 benchmark（结果没意义）
        exit(EXIT_FAILURE);
      }
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      try {
        // We don't reset dC between runs to save time
        run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
        // 先检查 kernel 启动错误（参数非法、资源不足等，同步，立即可知）
        cudaCheck(cudaGetLastError());
      } catch (const std::exception &e) {
        // 注意：此 catch 只能捕获 run_kernel 内 throw 的 C++ 异常（如非法 kernel 编号）
        // CUDA 错误由 cudaCheck 内部调用 exit() 处理，不经过异常机制，catch 捕获不到
        // benchmark 循环中捕获后仅打印不退出，循环继续但 elapsed_time 已包含异常帧的时间
        printf("%s\n", e.what());
      }
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    // cudaEventElapsedTime 返回毫秒，除以 1000 转换为秒
    // 1000. 是 double 字面量，elapsed_time(float) 会隐式提升为 double 再除，结果截断回 float
    // 严格写法应为 1000.f，全程 float 精度，避免隐式提升；此处精度差异可忽略
    elapsed_time /= 1000.;

    // floatPointOperations = 2*M*N*K：每个 C 元素需要 K 次乘法 + K 次加法 = 2K 次浮点运算，共 M*N 个元素
    // 用 long（64位）而非 int（32位）：
    //   m=n=k=4096 时，2*4096*4096*4096 ≈ 1.37×10¹¹，超出 int 上限（2.1×10⁹）
    //   用 int 会整数溢出，结果变为负数；long 上限 9.2×10¹⁸，可安全容纳
    //   m/n/k 也声明为 long，确保乘法运算全程在 long 精度下进行，不会中途溢出
    long floatPointOperations = 2 * m * n * k;

    // 列表初始化 {} 禁止窄化转换（narrowing conversion），这是它与 = 初始化的核心区别。
    //
    //   表达式的类型推导：
    //   (repeat_times * floatPointOperations * 1e-9) / elapsed_time
    //   //  int       *  long               * double  →  double
    //   //                                             / float → double
    //   // 最终结果是 double
    //
    //   double → long 是窄化转换（有精度损失），{} 直接拒绝，编译报错。
    //
    //   = 初始化允许窄化：
    //   long flops = (repeat_times * floatPointOperations * 1e-9) / elapsed_time;
    //   // double → long，隐式截断，编译通过但有精度损失
    double flops {(repeat_times * floatPointOperations * 1e-9) / elapsed_time};
    // %7.6f：float/double 格式，总宽7，小数6位（elapsed_time/repeat_times 为 float）
    // %7.1f：float/double 格式，总宽7，小数1位（flops 为 double）
    //   %f 对应 double 而非 float：printf 是可变参数函数，传入 float 会触发
    //   默认参数提升（default argument promotion），自动提升为 double
    //   因此 %f 和 %lf 在 printf 中等价（scanf 中必须区分，%f→float* %lf→double*）
    // %ld：long 类型整数（m 为 long）
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        flops, m);
    // fflush(stdout)：强制刷新标准输出缓冲区，将缓冲区内容立即写入终端
    // printf 默认行缓冲，数据可能积压在缓冲区，不立即显示
    // 每个 size 跑完后立即刷新，用户能实时看到每轮结果，而非等全部跑完才一次性输出
    fflush(stdout);

    // std::string("kernel") + "_" + argv[1] + "_" + "result.txt"
    // + 左结合，从左往右依次计算：
    //   std::string("kernel") + "_"      → std::string  （string + const char*，OK）
    //   结果                  + argv[1]  → std::string  （string + const char*，OK）
    //   结果                  + "_"      → std::string  （string + const char*，OK）
    //   结果                  + "result" → std::string  （string + const char*，OK）
    // 只需第一个操作数是 std::string，后续每次 + 左侧已是 std::string，右侧 const char* 自动转换
    // 若写 "kernel" + "_" 则是两个 const char* 相加，找不到 operator+，编译报错

    // 本质：a + b 是 a.operator+(b) 的语法糖，编译器自动转换，运算符重载均如此
    //
    // std::string::operator+ 的实现（简化）：
    //   成员函数版本（string + const char*）：
    //     std::string operator+(const char* rhs) const {
    //       std::string result = *this;   // 拷贝自身
    //       result.append(rhs);           // 追加右侧
    //       return result;
    //     }
    //   非成员函数版本（const char* + string，左侧是 const char* 时使用）：
    //     定义在 std 命名空间，不属于任何类，没有 this 指针，不是静态成员函数
    //     存在原因：成员函数 operator+ 只能处理 std::string + X（左侧必须是 std::string）
    //               无法处理 "literal" + std::string（左侧是 const char*），需要非成员重载覆盖
    //     直接用 lhs 构造结果再 append rhs，不是调换顺序后复用成员版本，两者是独立重载
    //     std::string operator+(const char* lhs, const std::string& rhs) {
    //       std::string result(lhs);   // 直接用 lhs 构造，不调换顺序
    //       result.append(rhs);
    //       return result;
    //     }
    const std::string resultDir = "benchmark_results";
    // std::filesystem::create_directories：递归创建目录，已存在则不报错（幂等）
    // 需要 C++17，CMakeLists.txt 中 set(CMAKE_CXX_STANDARD 17) 已满足
    std::filesystem::create_directories(resultDir);
    const std::string resultLogFile = resultDir + "/kernel" + argv[1] + "_result.txt";
    std::ofstream fs;
    // 第一个 size（SIZE 首元素）时覆盖旧文件，后续 size 追加写入同一文件
    // std::ios::app（append）：每次写入都定位到文件末尾，不覆盖已有内容
    // 不传 std::ios::app 时默认 std::ios::trunc，打开即清空文件
    if (m == SIZE[0]) {
      fs.open(resultLogFile);                        // 覆盖模式（trunc）
      fs << "Running kernel ";
      fs << kernel_num;
      fs << " on device ";
      fs << deviceIdx;
      fs << ".\n";

    } else {
      fs.open(resultLogFile, std::ios::app);         // 追加模式
    }

    fs << "dimensions(m=n=k) ";
    fs << m;  // 整型直接输出原始数字，std::fixed / std::setprecision 对整型无影响
    fs << std::fixed << std::setprecision(2);
    fs << ", alpha: ";
    // 此处尚未设置 std::fixed / std::setprecision，使用流的默认格式：
    //   格式：defaultfloat，自动选择定点或科学计数法中更紧凑的表示
    //   精度：有效数字 6 位（不是小数位），末尾零省略
    //   例：alpha=0.5 → 输出 "0.5"；alpha=0.000001 → 输出 "1e-06"
    // std::fixed / std::setprecision 一旦设置持久生效，直到显式重置
    // 此处写在设置之前，所以仍使用默认格式输出
    fs << alpha;
    fs << ", beta: ";
    fs << beta;
    fs << "\n";


    fs << std::fixed << std::setprecision(6);
    fs << "Average elapsed time: (";
    fs << elapsed_time / repeat_times;
    fs << ") s, performance: (";
    fs << std::setprecision(1);
    fs << flops;
    fs << ") GFLOPS. size: (";
    fs << m;
    fs << ").\n";


    // beta=3.0 时 kernel 执行 C = 0.5*A×B + 3.0*C，不是完全覆盖而是累加，初值直接影响结果
    // 下一个 size 验证时：
    //   dC_ref = 0.5*A×B + 3.0*dC_ref_old
    //   dC     = 0.5*A×B + 3.0*dC_污染值   ← 若不还原，dC_污染值≠dC_ref_old，验证必然失败
    // 还原后 dC==dC_ref，两者初值相同，验证结果才可比较
    // 若 beta=0 则初值无影响（C=0.5*A×B），无需还原
    // cudaMemcpyDeviceToDevice：全程在 GPU 内存间拷贝，不经过 CPU
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
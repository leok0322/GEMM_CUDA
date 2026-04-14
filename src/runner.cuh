// 【头文件中的声明可以重复，没有链接性问题】
// 声明只告诉编译器函数签名，不产生任何代码，可以被多个翻译单元重复包含
// 定义才产生实际代码，受 ODR（One Definition Rule）约束，只能有一份
//
// runner.cuh 被多个文件 #include 时，声明被重复展开，编译器只重新确认签名一致，无链接冲突
// #pragma once 防止的是同一翻译单元内的重复展开，跨翻译单元的重复声明本来就合法
//
// 对比：
//   声明（此文件）：void run_kernel(...);         → 可重复，无 ODR 约束
//   定义（runner.cu）：void run_kernel(...) {...}  → 只能有一份，受 ODR 约束
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>


void CudaDeviceInfo();    // print CUDA information

void range_init_matrix(float *mat, int N);
void randomize_matrix(float *mat, int N);
void zero_init_matrix(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
bool verify_matrix(float *mat1, float *mat2, int N);

float get_current_sec();                        // Get the current moment
float cpu_elapsed_time(float &beg, float &end); // Calculate time difference

void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle, int deviceIdx,bool record=false);
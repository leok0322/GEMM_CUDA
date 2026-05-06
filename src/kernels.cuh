#pragma once

#include "kernels/10_kernel_warptiling.cuh"
#include "kernels/11_kernel_double_buffering.cuh"
#include "kernels/12_kernel_double_buffering.cuh"
#include "kernels/1_naive.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_kernel_1D_blocktiling.cuh"
#include "kernels/5_kernel_2D_blocktiling.cuh"
#include "kernels/6_kernel_vectorize_and_resolve_As_bank_conflict.cuh"
#include "kernels/7_kernel_using_rearrangement_to_resolve_Bs_bank_conflicts.cuh"
#include "kernels/8_kernel_using_bank_extra_col_to_resolve_Bs_bank_conflict.cuh"
#include "kernels/9_kernel_autotuned.cuh"
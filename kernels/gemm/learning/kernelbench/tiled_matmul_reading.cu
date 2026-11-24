// 中文阅读版：基于 KernelBench 的 CUDA 源码，仅补充注释，计算与调用逻辑不变。
// 记号对应：输入矩阵 A[h,k]、B[k,w]，输出 C[h,w]；均按行连续存储。
// 注意：下面指针名 M/N 表示两个输入矩阵，不是通常 GEMM 记号中的维度 M/N。
// 本文件保留 PyTorch extension 的头文件和包装函数，不是独立的 main 程序。
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

// Macro to check if a tensor is a CUDA tensor
// 检查张量位于 CUDA 设备；TORCH_CHECK 在条件不满足时报告错误。
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

// Macro to check if a tensor is contiguous in memory
// 检查连续存储，保证下面可以用“行号 × 行宽 + 列号”直接计算元素下标。
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Macro to check both CUDA and contiguity requirements
// 合并上述两项检查。它们不等于完整的维度、精度或设备一致性验证。
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function for ceiling division
// 向上取整除法：例如 17 列、每块覆盖 16 列，需要 (17+15)/16 = 2 个块。
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}


constexpr int TILE_SIZE = 16;  // Size of each tile for matrix multiplication
// 每个线程块使用 16×16=256 个线程，对应输出矩阵的一个 16×16 小块。

/**
 * @brief Tiled matrix multiplication kernel.
 * 
 * This kernel performs matrix multiplication using shared memory tiles to improve performance.
 *
 * @param out Pointer to the output matrix
 * @param M Pointer to the first input matrix
 * @param N Pointer to the second input matrix
 * @param h Height of matrix M
 * @param w Width of matrix N
 * @param k Width of matrix M / Height of matrix N
 */
__global__ void tiled_matmul_kernel(float* out, float* M, float* N, int h, int w, int k) {
    // shared memory 属于一个线程块，块内线程共同装载、读取这两块数据。
    // M_tile 保存 A 的 [输出行, K方向] 小块；N_tile 保存 B 的 [K方向, 输出列] 小块。
    __shared__ float M_tile[TILE_SIZE][TILE_SIZE];  // Shared memory for M matrix tile
    __shared__ float N_tile[TILE_SIZE][TILE_SIZE];  // Shared memory for N matrix tile

    // Thread indices within a tile
    // ir/ic 是块内坐标：y 对应行，x 对应列，范围都是 0～15。
    int ir = threadIdx.y;
    int ic = threadIdx.x;

    // Global thread indices
    // blockIdx 定位输出块，threadIdx 定位块内元素；本线程负责输出 C[r,c]。
    // 如 blockIdx=(1,2)、threadIdx=(3,4)，则 c=1*16+3，r=2*16+4。
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    float res = 0.0f;  // Accumulator for dot product result
    // res 是本线程自己的累加值：最终等于 A 第 r 行与 B 第 c 列的点积。

    // Iterate over tiles
    // 把长度为 k 的点积分成若干段，每段最多 16 项；最后一段也向上取整。
    for (int K_tileidx = 0; K_tileidx < (k + TILE_SIZE -1) / TILE_SIZE; K_tileidx++) {
        // Load data into shared memory tiles, with bounds checking
        // 令 Kbase=K_tileidx*16，每个线程分别搬运一个 A 元素和一个 B 元素：
        //   M_tile[ir][ic] <- A[r, Kbase+ic]，全局元素下标 r*k+Kbase+ic。
        //   N_tile[ir][ic] <- B[Kbase+ir, c]，全局元素下标 (Kbase+ir)*w+c。
        // 两次装载的 K 下标分别使用 ic 和 ir，因为 A/B 小块的 K 方向不同。
        // 尾块或越界位置填 0，使后面的固定 16 项计算不会贡献无效数据。
        M_tile[ir][ic] = (((r < h) && (K_tileidx * TILE_SIZE + ic < k)) ? M[r * k + K_tileidx * TILE_SIZE + ic] : 0.f);
        N_tile[ir][ic] = ((((K_tileidx * TILE_SIZE + ir) < k) && (c < w)) ? N[(K_tileidx * TILE_SIZE + ir) * w + c] : 0.f);

        // 第一次块内同步：等所有线程完成装载，才能读取其他线程搬来的元素。
        // 即使本线程对应的 r/c 越界，也必须参与同步，不能提前 return。
        __syncthreads();  // Ensure all threads have loaded data before computation

        // Compute dot product for this tile
        // 本线程读取 A 小块第 ir 行、B 小块第 ic 列，累加当前 K 段的 16 项。
        // 同一个 A 元素可供块内 16 个输出列使用；B 元素可供 16 个输出行使用。
        // 这就是 shared memory 分块带来的块内数据复用。
        for (int idx = 0; idx < TILE_SIZE; idx++) {
           res += M_tile[ir][idx] * N_tile[idx][ic];
        }

        // 第二次块内同步：等所有线程读完当前小块，再覆盖 shared memory 装下一段。
        // 缺少这个同步，较快的线程可能覆盖较慢线程尚未读取的数据。
        __syncthreads();  // Ensure all computations are done before loading next tile
    }

    // Write result to global memory if within bounds
    // 全部 K 段累加后，只让有效线程写 C[r,c]；下标 r*w+c 的单位是 float 元素。
    // 若换算为字节地址，需要再乘 sizeof(float)，而不是在指针下标中手动乘 4。
    if ((r < h) && (c < w)) {
        out[r * w + c] = res;
    }
}

/**
 * @brief Wrapper function for tiled matrix multiplication kernel.
 * 
 * This function checks input tensors, sets up kernel parameters, and launches the CUDA kernel.
 *
 * @param m First input matrix
 * @param n Second input matrix
 * @return torch::Tensor Result of matrix multiplication
 */
torch::Tensor tiled_matmul_cuda(const torch::Tensor& m, const torch::Tensor& n) {
    // 这个包装函数在 CPU 上执行：取形状、分配输出、设置启动参数，再把工作提交给 GPU。
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);
    // m 的列数必须等于 n 的行数，点积长度才能一致。
    TORCH_CHECK(k==n.size(0), "Size mismatch");

    auto output = torch::empty({h, w}, m.options());
    // 输出沿用 m 的设备和 dtype；下面通过 data_ptr<float>() 按 FP32 访问数据。
    // empty 不初始化元素，正确的 kernel 必须覆盖所有有效输出。

    // Define thread block and grid dimensions
    // dim3 的顺序是 x/y/z：tpb.x 管列，tpb.y 管行，z 默认是 1。
    // grid 分别按输出列数 w、行数 h 向上取整，允许最后一块只覆盖部分有效元素。
    dim3 tpb(TILE_SIZE, TILE_SIZE);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));

    // Launch kernel
    // <<<网格中的块数, 每块的线程数>>>；这里未显式传 stream，使用默认 CUDA stream。
    // 启动通常是异步的：CPU 提交调用后，不代表 GPU 已经算完。
    tiled_matmul_kernel<<<blocks, tpb>>>(
        output.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(), h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // 上面的宏检查启动相关错误，不替代 GPU 完成同步或输出正确性验证。

    return output;
}

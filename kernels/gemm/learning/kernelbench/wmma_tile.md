# WMMA：一个 warp 共同计算 16×16 输出块

WMMA 步骤来自 CUDA Samples v13.0 commit `3f1c50965017932fc81e6d94a3fc9e04c105b312` 的 `Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu` 中的 `simple_wmma_gemm`。保留两次 fragment 装载、FP32 accumulator 清零、`mma_sync` 和行优先写回。将多 warp、多 K 段和 `alpha*AB+beta*C` 简化成一个 warp、一个 K 段和 `C=AB`。B 改为行优先，与前两课的物理布局一致。源文件、BSD 许可和 Git blob 哈希已归档。[固定版本源码](../../../../provenance/upstream-reviews/20261120-p1-01-wmma/cudaTensorCoreGemm.cu) · [来源记录](../../../../provenance/upstream-reviews/20261120-p1-01-wmma/source-record.json)

绑定和编译沿用 KernelBench commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` 的 `model_new_ex_tiled_matmul.py` 模式，复用 `load_inline`、PyTorch 分配和现有 CPU 参考。教学 wrapper 固定两个连续 `16×16` FP16 输入和 FP32 输出，核对 WMMA 所需的指针对齐。[CUDA 实现](wmma_tile.cu)

## 线程、地址与精度

启动 `<<<1,32>>>`。32 个 lane 都调用 `load_matrix_sync`、`mma_sync` 和 `store_matrix_sync`。每个 fragment 的元素分布由 CUDA 实现决定，代码通过 fragment 操作整块矩阵。A/B 的 `ldm=16` 是以元素计的行步长，各为 32 字节；FP32 输出的行步长是 64 字节。三张矩阵的起始地址须按 32 字节对齐。SM86 支持 FP16 输入和 FP32 累加的 `m16n16k16`。此例直接从全局内存装载 fragment。[CUDA 13.0 Warp Matrix Functions](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#warp-matrix-functions)

逻辑地址沿用 GEMM 公式：

```text
C[r,c] = Σ A[r,q] × B[q,c]，q = 0..15
A 字节偏移 = 2 × (16*r + q)
B 字节偏移 = 2 × (16*q + c)
C 字节偏移 = 4 × (16*r + c)
```

手算输入仅在 K=0、15 非零。令 `A[r,0]=r+1`、`A[r,15]=2`、`B[0,c]=c+1`、`B[15,c]=-1`，则 `C[r,c]=(r+1)(c+1)-2`，例如 `C[15,15]=254`。在 `q=15` 这一项，右下角用到 A 和 B 的最后一个半精度元素，各在对应基址后 510 字节；结果写在 C 基址后 1020 字节。

另用 `(1+2⁻¹²)×2048+1` 区分输入舍入与输出精度。A 写入 FP16 时，`1+2⁻¹²` 舍入为 1，因此 CPU 参考用存储值计算，期望 GPU 的 FP32 输出保留 2049。若再将结果转成 FP16，会舍入为 2048。

## 复现

在固定形状下运行手算块、密集带符号输入和舍入用例，各调用两次。保留实际 FP16 输入、每次 FP32 输出，使用 CPU FP64 点积复算，检查输入保持不变。

```bash
python3 experiments/p1_01/run_3070.py --case wmma \
  --evidence evidence/20261120-p1-01-wmma-replay
```

复现使用已授权的 3070 隧道和新证据目录。

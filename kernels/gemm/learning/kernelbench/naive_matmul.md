# 朴素 CUDA GEMM：每个线程从全局内存读取一条点积

来源为 KernelBench commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` 的 `src/kernelbench/prompts/few_shot/model_new_ex_tiled_matmul.py`，许可为 MIT。该例的 CUDA 实现使用 shared tile。为了单独观察逐线程的全局地址，从 `tiled_matmul_kernel` 和 `tiled_matmul_cuda` 简化出 [naive_matmul.cu](naive_matmul.cu)。保留 `16×16` 线程布局、行优先寻址、向上取整的网格、PyTorch 张量包装和 `load_inline` 编译方式。将协作装载和块同步换成线程独立的 K 循环，输出越界时可在读取前返回。现有验证入口负责 CPU FP64 参考和输入不变性检查。

## 从线程坐标算到字节偏移

`x` 对应输出列，`y` 对应输出行。设 A[M,K]、B[K,N]、C[M,N] 都按行连续存储：

```text
r = 16 * blockIdx.y + threadIdx.y
c = 16 * blockIdx.x + threadIdx.x
C[r,c] = Σ A[r,q] * B[q,c]，q 从 0 到 K-1
A 元素偏移 = r*K+q
B 元素偏移 = q*N+c
C 元素偏移 = r*N+c
```

例如 `M=17,N=19,K=7` 时，块 `(x,y)=(1,1)` 中的线程 `(2,0)` 负责 `C[16,18]`。当 `q=6` 时，A、B 分别读元素偏移 118、132，累加结束后写 C 的元素偏移 322。FP32 每元素 4 字节，对应各自基址加 472、528、1288 字节。块内线程 `(3,0)` 对应列 19，在读 A/B 前返回。

固定 `q` 时，同一输出行内相邻的 `x` 线程读取同一个 A 元素和相邻的 B 元素。每个输出线程独自读取 A/B、累加和写回，所以没有跨线程装载依赖。满 16×16 输出块中，同一 A 元素在源码层面会被 16 个输出线程分别读取，B 同理。真实显存流量还受访存合并与缓存影响。[对照 shared tile 的协作装载](shared_tile.md)

## 复现

固定 FP32、二维、正维度、连续行优先、同一 CUDA 设备、默认 stream。用地址演示矩阵及三个小整数矩形输入检查数值，覆盖双轴尾块、单行和单列。每组调用两次，对每个输出逐元素比较，并检查 A/B。

```bash
python3 experiments/p1_01/run_3070.py --case naive \
  --evidence evidence/20261120-p1-01-naive-replay
```

复现使用已授权的 3070 隧道和新证据目录。

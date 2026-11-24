# P1-01：先看一个线程对应哪个输出

2026-11-19。来源为 KernelBench 固定 commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` 的 `src/kernelbench/prompts/few_shot/model_new_ex_tiled_matmul.py`。来源／hash／提取方式见 [SOURCE.json](SOURCE.json)，MIT 许可见 [LICENSE.kernelbench](LICENSE.kernelbench)。

2026-11-20 已沿用上游原始 ModelNew/load_inline 在 RTX 3070 编译并执行，14 组小规模 FP32 输入各 2 次，输出精确一致且输入不变。启用 Windows WDDM 调试接口后，原二进制的 memcheck/synccheck 均通过。P1-01 整项仍待学习验收。[原始结果](../../../../evidence/20261120-p1-01/REPORT.md) · [Sanitizer 复验](../../../../evidence/20261120-p1-01-sanitizer-repair/REPORT.md)

接续练习用 `M=N=2, K=17` 验证两段 K 的贡献。先看越界输出线程如何参与装载，再核对尾块填零和两个屏障。[学习与自测](shared_tile.md) · [3070 实测](../../../../evidence/20261120-p1-01-shared-tail/REPORT.md)

朴素版沿用相同的输出线程布局，每个线程独立从全局内存读一条点积。3070 上已通过双轴尾块、单行、单列和地址演示四组输入。[CUDA 与地址推导](naive_matmul.md) · [3070 实测](../../../../evidence/20261120-p1-01-naive/REPORT.md)

WMMA 版改编自固定 NVIDIA CUDA Samples，又沿用同一 load_inline 编译方式。32 个 lane 共同算一个 16×16 输出块，已在 3070 验证 FP16 输入和 FP32 输出。[CUDA 与精度推导](wmma_tile.md) · [3070 实测](../../../../evidence/20261120-p1-01-wmma/REPORT.md)

## 阅读哪份文件

- [tiled_matmul_reading.cu](tiled_matmul_reading.cu)：从上游 `source` 字符串提取，2026-11-20 仅增加中文注释；已核对非注释 token 与原文一致，kernel 与 wrapper 算法未修改。
- [model_new_ex_tiled_matmul.py](model_new_ex_tiled_matmul.py)：上游原文件，保留其 PyTorch `load_inline` 入口。未来优先通过这个成熟扩展入口验证，不引入 P0-05 自定义 ABI 或加载器。

`.cu` 是阅读视图，不代表已经实现新的独立 nvcc 构建入口；它使用 PyTorch extension 头和 wrapper。

## 本次只理解线程与地址

采用通常的维度记号：A[M,K]、B[K,N]、C[M,N]。本例源码的指针名 `M`/`N` 分别代表 A/B，标量 `h`/`w`/`k` 分别对应矩阵维度 M/N/K，不要把名字混淆。

```text
C[r,c] = sum_k A[r,k] * B[k,c]
A 的一维下标 = r*K+k
B 的一维下标 = k*N+c
C 的一维下标 = r*N+c
```

FP32 的字节地址再乘 sizeof(float)=4；以上是一维元素下标。

上游的线程映射：

```cpp
int r = blockIdx.y * blockDim.y + threadIdx.y;
int c = blockIdx.x * blockDim.x + threadIdx.x;
```

启动线程块为 `16×16`，x 对应输出列、y 对应输出行。每个有效线程负责一个 C[r,c]，沿 K 方向累加。沿用这个输出归属，可以跟踪块内线程如何装载和复用 A/B。[装载和同步示例](shared_tile.md)

检查题：A 为 2×3、B 为 3×4，按行连续存储；计算 C[1,2] 时，k=0/1/2 分别读 A、B 哪些一维下标？

2026-11-20 用户回答“A的第1行和B的第2行”。A 的行选择正确，B 的行／列仍有混淆。已解释：应取 A 第1行和 B 第2列（均从0开始），A 一维下标为 3/4/5，B 为 2/6/10。r 与 c 固定，k 变化。后续 C[0,1] 检查题，用户回答“0和1”，正确对应 A 第0行／B 第1列，已纠正基础选择；完整 tile 手算与 shared 同步仍需后续学习验收。

## 已实测范围与剩余边界

本次只在 FP32、二维连续、同一个 CUDA 设备、正维度和默认 stream 下验证；小形状同时包含 16 的整倍数与尾块。上游 wrapper 没有完整 dtype/rank/device 范围检查，launch 未显式传当前 stream，因此不能直接宣称通用生产接口。

既有 nvcc、C++、ninja 已满足该入口，未安装／升级软件。CPU FP64 参考和 torch.testing.assert_close 用于精确检查输出及输入不变性，编译日志和产物 hash 已保存。Sanitizer 的工具错误与程序的正确性结果分别记录。P1-01 仍需 tile 手算与同步自测。

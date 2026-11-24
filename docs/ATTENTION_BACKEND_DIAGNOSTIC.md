# Attention 后端诊断（P2-04）

2026-11-22。两卡分别用3个独立Python进程，强制实际backend，prefill与decode分列；每后端3 warmups、10 trials、discard_first=1，原KernelBench冷L2/CUDA Event函数逐字导入。两卡各24行/240原始样本，先/后FP64正确性及输入不变检查通过。

这不是配对晋升或agent收益实验；不替代正式A/A/P6。下表给全部3个session的中位数（微秒），不删除高值。3070 session0的prefill fa2与SDPA math明显高于后两次，说明当前单次冷cache诊断受初始化/时钟/测量环境影响，不能据此写稳定加速比。

| 阶段/后端 | 3070 s0 / s1 / s2 (µs) | 4090 s0 / s1 / s2 (µs) |
|---|---:|---:|
| decode/decomposed-fp32 | 43.008 / 41.984 / 43.008 | 32.768 / 32.768 / 32.768 |
| decode/flashinfer-fa2 | 17.408 / 17.408 / 17.408 | 15.360 / 14.336 / 15.360 |
| decode/torch-sdpa-flash | 19.568 / 20.384 / 20.432 | 15.360 / 15.360 / 16.384 |
| decode/torch-sdpa-math | 72.240 / 72.192 / 72.208 | 46.080 / 47.104 / 46.080 |
| prefill/decomposed-fp32 | 70.656 / 69.632 / 69.632 | 53.760 / 54.272 / 54.272 |
| prefill/flashinfer-fa2 | 87.552 / 14.336 / 13.840 | 11.776 / 12.288 / 11.264 |
| prefill/torch-sdpa-flash | 13.312 / 12.288 / 13.312 | 10.240 / 10.240 / 10.240 |
| prefill/torch-sdpa-math | 493.568 / 101.376 / 101.376 | 78.336 / 82.944 / 71.168 |

prefill：B1/Sq128/Sk128/H4/D64，causal。decode：B1/Sq1/Sk1024/H8/D64，全部有效KV可见。FP16输入/输出，CPU参考从同一量化输入以FP64计算，atol/rtol均2e-3，TF32关闭。

分解版用FP32 QK、stable softmax、PV，物化完整score；SDPA MATH亦为分解实现。FlashInfer fa2和强制SDPA FLASH走成熟CUDA融合路径，不物化完整score。计时排除H2D、CPU参考及JIT/plan，包含函数内cast、布局转换和输出copy。SDPA FLASH由强制context选中，失败不会静默降级；实际两卡均可用。单次Event保留上游cold-cache/host launch边界，不能将它混为长段Graph性能。

3070：Torch2.14/CUDA13，FlashInfer0.2.8仅C++20编译标志适配。4090：Torch2.7.1/CUDA12.6，FlashInfer0.2.8原C++17。各卡结果只描述该软件/硬件环境，不能以跨卡绝对差异证明agent迁移收益。

首个3070入口在计时前因WSL nvidia-smi PATH缺失退出（0样本），原失败保留；补回已知路径后才得到本批数据。无新API、无候选晋升、生产状态不变。

[3070全部原始结果](../evidence/20261122-attention-backends-sm86-fixed-smi-path/result.json) · [3070审计](../evidence/20261122-attention-backends-sm86-fixed-smi-path/verification.json) · [4090全部原始结果](../evidence/20261122-attention-backends-sm89/result.json) · [4090审计](../evidence/20261122-attention-backends-sm89/verification.json)

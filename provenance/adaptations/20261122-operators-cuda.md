# P2 / P4 固定 CUDA 接入（实施前记录）

- PyTorch pin 为 `e2d141dbde55c2a4370fac5165b0561b6af4798b`（v2.7.1），以下载 ref.json 为准。采用 `aten/src/ATen/native/cuda/PersistentSoftmax.cuh::dispatch_softmax_forward`，原文件及 BSD 许可原样保留。薄 CUDA/Torch 绑定仅检查 layout/dtype、分配 FP32 输出、设置 device guard 和调用原函数；宽度大于 2048 使用 ATen 原生 CUDA softmax。没有新设计归约 kernel。
- FlashInfer v0.2.8 `3f8317c67ac129facb8525b2fb581be42f634459`：采用 `flashinfer/prefill.py::single_prefill_with_kv_cache` 的 fa2 CUDA backend；`flashinfer/mla.py::BatchMLAPagedAttentionWrapper` 和 `include/flashinfer/attention/mla.cuh` 的 fa2 分页/自动 split-KV 路径。保留原调度、JIT、归约和 CUDA kernel，不移植到 Triton。Apache-2.0 许可原样保留。原下载失败记录 source-record.json / fetch-v2.json 保留，curl 成功文件以 fetch-v3.json 为准。
- `mla.cuh` 的 `final_lse=ptx_log2(d)+m` 是 log2；适配器明确转换为自然对数。核心输出是 latent Z，完整层还需 head up-projection。scale 来源是吸收前的 D_nope+D_rope，不能按 latent R+D_rope 重算。
- FlashMLA 计划中历史 pin `ba89a3466e9470ad08ab39738d4e7bb66989e1e7` 要求 SM90/SM100；现有 SM86/SM89 不运行该专家对照，不编造成绩。
- 必要新增：合同和输入布局适配、可信 CPU FP64 参考、empty KV 的显式零输出/负无穷 LSE 语义、原始证据。CPU 参考不进入候选 worker。参考原型先完成，5 项 CPU 数学检查已通过，不能当 CUDA 验收。

预注册验证：`python -m unittest discover -s tests -p test_operator_references.py -v`；CUDA 接入在独立目录运行公开 `operator_validate.py`，保存源码、环境、编译日志和逐项结果。Softmax FP32 atol=2e-6 rtol=2e-5；Attention/MLA FP16 输出 atol=2e-3 rtol=2e-3，MLA 自然 LSE atol=2e-3 rtol=2e-3。独立 FP64 参考以实际量化输入为起点。近零元素逐元素使用绝对项，不用整体 norm 隐藏错误。timing 和 profile 分开，GPU 忙时只做构建和正确性。

服务器只执行这些已审查固定源码/上游库；namespace/Docker 隔离尚未验收，不执行任意模型源码。

后续最小适配（写入后实施）：同一 FlashInfer pin 的 `rope.py::apply_rope_pos_ids`（显式 interleave=True）、`page.py::append_paged_mla_kv_cache`，以及 `cascade.py::merge_state` / `attention/state.cuh` 的 log2 LSE 合并。分页位置由可信控制侧计算，不采用该 SDK 中可选的 Triton positions helper。split 合并对两侧空行先用输入长度分派，避免上游 `-inf-(-inf)`；非空行仍调用原 CUDA merge kernel，计时包含 combine。原 SDK 内部自动 split 与显式分片对照分别报告。

3070实测缺口：原SDK成功import，但Torch2.14头文件的std::strong_ordering在SDK固定C++17下编译失败。原40项检查/错误留档在`evidence/20261122-attention-mla-sm86/`（Softmax23通过，其余17失败）。后续在**新prefix副本**仅将`flashinfer/jit/core.py`两处`-std=c++17`改为`-std=c++20`，原文件SHA`3f160c9876c27ffbe1444081c4c87439d17d43d205a87cb44609310d1168184e`，CUDA算法、API、容差和原prefix逐字保留。使用新项目独立JIT cache并归档，不与其他Torch版本共用cache。服务器Torch2.7继续原始SDK标志。验证命令仍为公开operator_validate，全部失败保留，不触及封存GEMM清单。

后续guard适配：SDK固定CUDA路径的支持合同收紧为FP16指针16字节对齐（未对齐在host拒绝，不尝试危险native调用）。显式custom mask与bottom-right causal取交集，bool SQ×SK同设备；全mask行在原SDK输出后用ATen where显式归零。该适配仅输入合同/空行语义，不改CUDA算法；公开offset8对齐/offset1拒绝/全mask和Sanitizer另做delta验收。原40项验证source快照不改。

P2-04诊断计时（实施前）：直接导入所锁KernelBench `timing.py::time_execution_with_cuda_event` 原文件，不改冷L2清理/Event实现。对比FP32分解QK-softmax-PV、SDK fa2融合、强制Torch SDPA MATH/FLASH；统一FP16输入/输出，TF32关闭。prefill与decode分开，3新进程，各3warmups/10trials/1discard，保留全部原始毫秒。先检查3次0%空闲，忙则不采样；先CPU FP64正确性再计时。排除H2D/参考/JIT，包含cast/layout/output copy；方法含原上游单次Event及cache flush边界的限制。此为描述性冷cache后端诊断，不是配对晋升/agent收益，不替代正式A/A或P6三session配对。

# Softmax / Attention / MLA 接入边界

## 当前实测

Softmax 的 PyTorch v2.7.1 原生 CUDA header 薄绑定已在 RTX3070、Torch2.14/cu130 下构建并正确运行。22项公开输入、三模式各16项offset/stream检查及4项非法输入拒绝通过；memcheck/synccheck 0errors。该结果不等于4090或发布包已验收。[证据](../evidence/20261122-softmax-sm86/verification.json)

CPU FP64 的稳定softmax、online state、attention、RoPE、MLA显式/吸收参考5项检查通过。Attention/MLA固定FlashInfer CUDA适配已在3070通过40项公开检查（含23项Softmax）；后续offset/stream/全mask/页表/完整层projection三模式检查及memcheck、synccheck均0errors。[GPU证据](../evidence/20261122-attention-mla-sm86-cxx20/verification.json) · [guard证据](../evidence/20261122-attention-mla-sm86-guards/verification.json)。4090/完整workload/性能/部署仍待验收。

## Softmax

CUDA contiguous rank2，FP16/FP32输入，FP32输出；有限值绝对值≤10000。行数1..65536、宽度1..16384，调用方应在自身显存预算内提供张量。宽度≤2048调用原 `dispatch_softmax_forward`；更宽走ATen CUDA softmax。空维度、非连续、错误dtype/rank在host侧拒绝。当前公开实测行数5/7及宽度1/7/31/32/127/129/513/1024/2048/2049/4097，不能将有限测试冒充全部枚举。容差atol2e-6、rtol2e-5。

## Attention

CUDA FP16 contiguous、16字节对齐 BSHD，head dimension64/128，当前head数一致，query长度不大于KV长度。输出FP16。fa2融合路径由固定FlashInfer CUDA实现；batch适配暂逐请求调用，开销计入端到端测量。causal按bottom-right对齐，显式mask以True表示可见，与causal取交集，全mask行输出零。FP64参考从实际FP16输入开始，逐元素atol2e-3、rtol2e-3。部署/性能范围由实际GPU结果决定。

## MLA

经典dense、Sq=1、R512/Drope64、FP16输入及latent Z输出，LSE FP32且统一自然对数。使用原生fa2分页/split planner，page_size1/16/64，heads16/32/64/128；空KV请求返回Z=0/LSE=-inf。索引和长度为可信host整数，页表越界和跨请求重复页在适配器规划前拒绝。

`q_abs=Q_nope U_K^T`，`score=(q_abs C^T + q_rope K_rope^T)*scale`，`Z=softmax(score) C`，完整层输出还需 `Z U_V`。scale按吸收前D_nope+Drope计算（V3代表维度为1/sqrt(192)），不能用R512替换D_nope128。CPU全层吸收等价与FP16 native core精度分开验收，不能把q_abs的重新量化误差藏进数学等价。

RoPE用明确token位置、interleave=True，query与位置key相同预处理；cache append接收已经做完位置处理的k_rope，不隐式重复旋转。append输入按请求分组，位置为old_length..new_length-1，host验证final页表和tail，保留旧映射由调用方保证。native core不物化完整score/probability；CPU参考可以物化，且不进入候选worker。

SDK原始LSE是log2；适配器乘ln2。显式split合并用原CUDA `merge_state`，先把自然LSE转换回log2；两侧空行按length分派避免-inf相减，combine必须纳入计时。128MiB workspace与plan成本分开报告。

来源、保留部分和修改说明：[实施前采用记录](../provenance/adaptations/20261122-operators-cuda.md)。当前服务器缺少已验收的候选隔离，只运行已审查源码及成熟SDK。

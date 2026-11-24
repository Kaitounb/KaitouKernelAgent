# P3-01 首轮源码提案草案

2026-11-22。P3-02受限builder／GPU worker闭环已通过；真实源码提案尚未调用。复用已验证的DeepSeek KernelAgent provider，单次最多768输出tokens、60秒、零重试。

从当前4-warps WMMA实现出发，提出一次CUDA源码结构变化：K循环一次处理两片16-wide tile，同时保留单片尾部、相同FP32累加次序，不加barrier、不改host launcher。预期仅是假设，可能无收益或退化。真实候选必须通过源码审查、构建、完整合同正确性及新计时裁决；旧参数结果不能代验。

模型只返回单文件unified diff及reason。固定license/includes、host dispatcher、cuBLAS回退、合同、参考、容差、测量规则均不允许修改。封存清单／参考／凭据不提供。request与本地parent绑定已在evidence/20261122-p3-01-source-prepared/，status为PREPARED_NOT_SENT。

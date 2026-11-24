# P1-03 固定 CUDA 模板参数裁决草案

2026-11-21，实施前草案。按 KDA 固定版本的 draft→plan→candidate→correctness→measurement→evidence/promotion 推进。

当前参照为 P1-02 4-warps WMMA 模板；合法配置仅 1/2/4/8，尾块／非对齐使用 cuBLAS。复用已验收 KernelBench 编译产物和 `evaluators.gemm_regression::check_case`，不执行模型生成 Python/CUDA。冻结数值合同与测量门槛不变。独立清单不读取、不传输。

首先实现严格参数 JSON、parent／源码／合同／评测器 hash 绑定；再把已有 A/A Graph/Event 循环适配为同一输入、双函数、三进程、两缓存策略 A/B。评测器根据完整开发集、正确性、原始段和冻结门槛裁决。错误、超时、不完整证据和无收益保留当前参照；初始 CUDA 实现与可信 cuBLAS 回退分开记录。

风险：真实 API 服务／模型／预算尚未配置；不能把静态 fixture 或本会话推断当真实外部提案。GPU 忙时不做正式性能裁决。已有约 1% 图间偏差保留并采用已冻结保守门槛。现阶段只跑已审查参数与可信故障；完整隔离待 P3-02。

首轮分两部分：可信 fixture 证明输入／输出错误和主机超时不会改 incumbent，以及同实现 A/A 无收益回退；有模型配置后，一次实际 API 返回合法参数、评测并留档，无需获得晋升才算真实闭环。模型凭据只在本机提案进程，GPU 侧环境使用 allowlist。候选侧没有源码写入通道。

验证：CPU 提案／证据／门槛负例测试；3070 上完整开发和边界正确性、3 session × 2 policies × 8 workload 的原始配对段；真实模型 request/response/usage 与拒绝理由；轨迹重放计算 incumbent。模型配置缺失时完成其余工作并将真实调用验收保留未完成。

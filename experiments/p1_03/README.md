# P1-03 首轮固定参数裁决

继续复用 P1-02 的可信 WMMA/cuBLAS 二进制和 KernelBench 路径。模型只能交出 `schema_version/parent_id/warps/reason` JSON；1/2/4/8 之外、bool/float、未知字段、重复 JSON key、非有限常量与过期 parent 都拒绝。reason 不作为可执行内容。

当前入口是一次新轨迹的首轮：初始 incumbent 为已验收 4 warps。不是持续多轮搜索服务；后续持续搜索必须显式继承最后状态，不能把每轮新目录当作自动重置当前参照。KDA 工作流与 ForgeKernel 当前参照／评测保护取舍见 [实施前计划](../../docs/p1_03/plan.md)。

## 真实提案

本机缺少模型 SDK，已独立安装到临时 venv，未改 CUDA 环境。后续可用锁定依赖创建同等隔离环境：

```text
python3 -m venv <model-venv>
<model-venv>/bin/python -m pip install -r experiments/p1_03/requirements-model-lock.txt
DEEPSEEK_API_KEY=<通过私有控制层环境注入，勿填进命令历史>
<model-venv>/bin/python -m experiments.p1_03.propose --output <new-proposal-dir>
```

`propose` 仅通过固定 KernelAgent `OpenAICompatibleProvider` 调用 DeepSeek，max_tokens=768，timeout=60s，零自动重试。每个新目录最多一次请求；失败也留 usage，不自动重试。密钥只从环境读取，原始请求不含密钥。首次仅发送公开 NVIDIA WMMA 示例和硬件、合法参数，`current` 别名在本地绑定 parent hash；未发送源码、合同、开发输入或独立清单。用户已允许后续优化按需提交相关源码／开发数据，独立清单仍按既有冻结协议持有。

真实返回、解析后绑定的 proposal、resolved model/request ID/usage 分别留档。空响应、坏 JSON、非法参数均结束为拒绝，不替模型修代码、不追加付费调用。首次记录 431 tokens，未获响应的真实账单字段，不将估价当实付费用。

## GPU 与裁决

```text
python3 -m experiments.p1_03.run_3070 --proposal <proposal.json> --evidence <new-evaluation-dir>
python3 -m experiments.p1_03.replay --evidence <evaluation-dir> --output <new-replay.json>
python3 -m unittest discover -s tests -p test_gemm_decision.py
```

复用 HANDOFF 中已授权 21022 转发和 P1-02 远端原二进制／依赖；丢失时先核查，不安装整套框架。原始文件和 flags/hash 在 P1-02 证据中，参数候选不重复编译。GPU 子进程环境 allowlist，不传模型密钥或独立清单。

每个独立 session：8 开发 × 2 seeds + 6 类边界 × 2 seeds 的新鲜测前／后正确性；8 workload × 两缓存策略的候选自身 A/A 和 incumbent/candidate A/B；各 9 对、32 Graph 节点 × 12 replay，捕获／测量顺序有记录，输出及输入快照再核对。三 session 全量原始段外置裁决。正式性能只在 GPU 空闲窗口运行；profiler 不参与。

`gemm_decision.py` 从原始段重算会话级区间和冻结门槛，缺失／篡改／错误／A/A 不稳即拒绝。只有全部条件通过才原子更新本轮 incumbent.json，JSONL 保存故障、恢复、实际候选和 parent。`replay` 重新计算 verdict 与 final state，不调用模型或 GPU。

四个可信故障（常数、输入污染、主机异常、有限 CUDA 后 host 超时）经过同一监督入口，所有拒绝和新进程恢复留档。本阶段依然是可信参数／可信夹具、同 UID/root GPU 进程，未验收 P3-02，也不自动执行模型源码。

# P1-03 执行计划与上游采用

2026-11-21。实现前记录。预算与 API 配置从控制层传入，禁止从模型输出覆盖。

| 固定上游／符号 | 保留与采用 | 必要改动／具体缺口 |
|---|---|---|
| KDA `acdd79343c5e9d0c7cb96289f26fd3f1051ce785` `docs/agent-flow.md`, `prompts/basic-flow.md` | draft/plan→候选→正确性→测量→保留／拒绝；简单 candidates.jsonl 和 parent 链 | KDA 为工作流文档，无此 SM86 模板可直接执行的裁决 SDK；补合法 warp JSON、产物 hash 和已冻结门槛适配 |
| ForgeKernel `b2c8ae8510e02c28576808910a6b8827721ab7dc` `src/forgekernel/loop/harness_guard.py`, `src/forgekernel/window_contract.py` | 可信评测副本、当前参照、双边相同 timed-window、raw ratio 决策，不折算税后比值 | 原 guard 依赖 Git lineage 与 card，当前项目无 Git revision；采用 staged 文件 SHA256 绑定，不宣称复用其 Git revert 实现或完成权限隔离 |
| KernelBench `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` 原 loader/correctness + 已验收 P1-02 artifact | 复用现成绑定、CUDA `.so`、独立 CPU FP64/不可变/guard/stream 校验；不重建 FFI | 参数候选调用同 `.so` 中 `gemm_template(a,b,warps)`，构建证据复用明确 hash；补全 A/B 当前 incumbent 对照与新鲜测前／后校验 |
| 项目 `experiments/p1_02/calibrate_gemm_aa.py` | 32 Graph 节点 × 12 replay、9 对、交替顺序、1/16 输入、原始 Event 段 | 将同函数 A/A 变为两个合法模板函数；独立 session 交替 capture 次序；覆盖全部 8 开发保护 workload |
| KernelAgent `e0647170da36ef9b059ac0bd3d60103aa4ed378b` `utils/providers/openai_base.py::OpenAICompatibleProvider`、`base.py::LLMResponse` | 优先复用已有 OpenAI 兼容 provider 调用／usage 接口，先核对完整依赖和实际配置 | 原 provider 有隐式高 reasoning／tokens 上限与默认 SDK 重试；有配置后采用有界 timeout／零自动重试，保留必要参数与 usage；未获配置不付费调用 |

执行顺序：严格提案和纯 CPU 门槛 → 可信二进制 worker 配对与错误夹具 → 本机监督及不可变轨迹 → 模型 provider 接入 → 原始证据／CPU 重放／TASKS 更新。

预定命令：

```text
python3 -m unittest discover -s tests -p 'test_gemm_decision.py'
python -m evaluators.gemm_candidate --binary <trusted.so> --sha256 <known> --warps <1|2|4|8> --session <0|1|2> --output <new.json>
python3 -m experiments.p1_03.run_3070 --evidence <new-dir> --proposal <bounded.json>
python3 -m experiments.p1_03.propose --config <local-private-config> --output <new-dir>
```

独立 workload 清单不在任何上述输入中；模型只接收开发合同、可信模板、合法参数与当前参照。P3-02 前禁止自动运行模型源码。模拟门槛、可信夹具、GPU 实测和真实外部 API 调用分别标记，不用前者代替后者。

用户已指定 DeepSeek 和本机 Pages 中 `deepseek_mac_claude` 密钥。首轮 1 次调用、max_tokens=768、非 thinking、零自动重试。自动审批拒绝了完整源码／合同／开发集外传，改为仅公开 NVIDIA WMMA 描述、公开硬件和合法参数；原始 parent hash 由控制侧以 `current` 别名本地绑定，源码／合同／workload／参考与晋升规则均不发送。真实候选仍用原冻结合同独立评估。

用户随后明确授权相关源码／开发数据可交给模型。已提交的首轮保留最小公开请求，不追加调用；独立验收清单继续按冻结协议保留。首轮实际 `deepseek-flash` 返回 warps=8，339 input + 92 output = 431 tokens，只有一次网络调用，无自动重试。模型理由只作为假设，不能替代 GPU 证据。

真实 8-warps 候选的三进程正确性通过，A/A 在 session 2/dev-04/repeated 出现 25.3012% 单对离群，按原冻结界限拒绝；该裁决不重跑替换、不放宽门槛。追加独立可信 4-vs-4 负对照，固定 3 session、同 8 workload／两缓存／原全部检查，只验证无收益不晋升，不改变实际候选裁决或模型预算。复用同评测源码／二进制，已完成的故障组不重复运行。若该对照也不稳定，保留其拒绝与明确限制。

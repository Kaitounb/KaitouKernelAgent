# 旧本地材料审查与当前采用边界

日期：2026-11-18，Asia/Shanghai。审查对象是迁移快照，不重新执行旧 NPU 测试。原快照 61 个文件逐项与 MIGRATION.json 中的 hash 相同；本次没有修改 references/。

本文核对 P0-05 的旧本地材料及其当前用途；五个主要上游另见 [UPSTREAM_REUSE.md](UPSTREAM_REUSE.md)。当前工程以成熟组件复用／适配为主，旧 NPU 材料仅作经验参考。原 P0-05 报告与源码快照保留历史设计，本文的当前决定不再要求独立重写 CUDA 框架。

## 采用决策

| 来源模块／文件 | 可借鉴机制 | 假设或差异 | 本次决策 |
|---|---|---|---|
| `ascend_optimizer/engine.py` | task/source hash、顺序验证、原子证据、fresh session、当前 incumbent 配对、最终对最初基线复测 | 默认 npu:0、`.py` 模板渲染、短候选 ID、source-only 身份；缺 CUDA 编译／产物／头依赖绑定 | 保留历史经验；工程优先采用 KDA／ForgeKernel 等现成候选／证据流程，仅补任务连接 |
| `statistics.py` | 拒绝非有限／非正时间、保留配对、按 workload 等权汇总 | 每 workload 用配对比值中位数，旧 bootstrap／阈值与本项目配对 log-geomean 协议不同 | 不沿用旧实现／阈值；先核对成熟统计工具，必要时只适配已冻结的本项目筛选协议 |
| `tasks.py` | 固定 schema、输入语义、验证配置、规范化 JSON hash | 导入 GDN specs；只含 NPU vector-add/RMSNorm/GDN，Triton 模板；block 参数不能当 CUDA threads | 旧模板不移植；优先对齐 FlashInfer-Bench／KernelBench 定义，已有 C++ 元数据仅为原型 |
| `runtime.py` | 超时、独立进程组、输出上限、凭据过滤 | 持久 Python NPU worker、POSIX select、黑名单过滤不等于最小环境；同 UID 无隔离 | 优先使用选定上游 runner／进程管理，移植必要超时场景；实际隔离仍待 P3-02 验证 |
| `llm.py` | 与硬件解耦、API 参数约束、usage、重定向处理与脱敏思路 | 仍带 ASCEND_* 配置；供应商协议／usage 字段需实际接口验证，不能预称 CUDA 可用 | 当前未复制／调用；P1-03 优先用成熟模型适配器，核对实际已授权接口后仅补缺项 |
| `agent.py`、`cli.py` | 工具顺序、候选预算、必须做明确裁决 | NPU prompt、Triton 配置 schema、旧命令名和包依赖 | 先采用成熟工具／工作流，再做 CUDA 任务 schema 的必要适配；不能只替换设备字符串 |
| `worker.py` | 量化后 CPU reference、输入／guard／NaN 检查、benchmark 前重验 | torch_npu、NPU Events、msprof；ABBA/BAAB 与 P0-03 的 9 对顺序不同；基线/候选复用同组 args；没有 P0-04 式测速后复检 | 优先复用 KernelBench／FlashInfer-Bench 评测和 KernelAgent profile；必要时补双侧 buffer／前后检查 |
| `gdn_templates.py`、`gdn_evaluator.py`、`project_sources/` | 项目源码 pin、路径与状态语义记录 | GDN/NPU 专用，且不属于 GEMM/经典 MLA 主线 | 全部留作参考，不移植／运行 |
| 旧 `tests/` | 过期 benchmark、源码污染、无正确性不晋升、工具调用等故障场景 | 含 FakeWorker／mock，不能证明 NVIDIA 实卡行为 | 借鉴场景，CPU／模拟／GPU 结果分别留档；本次不重跑旧测试 |
| `experiments/p0_03` | 合同、Graph/Event 计时边界、原始数据与独立复算 | Triton/PyTorch 路径，噪声不推广到 C++ 或其他 shape | 原样保存；Cuda 路径接入后另验 |
| `experiments/p0_04` | 输入污染／缓存结果／尾部／延迟故障、进程组清理、best 写入门 | 单个初始 baseline 对照；不是多轮 incumbent 搜索；同 UID/root | 原样保存；复用测试要求，不整包充当主工程 |
| `references/learning/*` | 既有地址、WMMA、Attention 数学与例子 | 本地教学来源的完整许可证／revision 未均具备；README 多处记录仅静态检查 | 阅读用途；本次不复制到 active kernels，不标已 GPU 验证 |

**本次没有把任何旧 Ascend 模块或第三方 kernel 复制／导入到主实现。** P0-05 新增的 C++ 元数据、接口头和主机测试只保留为教学／设计原型；不作为默认工程 ABI。Python API 适配没有因“可复用”就计为已实现。

## 来源与许可核对

1. Ascend optimizer 是本地工作区快照；其目录与父目录都没有可用 Git revision。以逐文件 SHA256／快照 tree hash 定位，不伪造 commit。原 NOTICE 声明其 orchestration/evaluator 为该工作区独立实现，但没有为整个 optimizer 声明通用许可证；不能把 GDN 的 Apache-2.0 扩大到整个包。本次采用 ideas-only，不分发改编包。
2. 两份 GDN 源码保留 Apache-2.0 SPDX。其 origin.json 记录 commit `622725cd44aa47186dc1f1c9f08485e9013fff7a`、原路径与文件 hash；本次在已有本地 upstream checkout 用 `git show <commit>:<path>` 对照，两个文件逐字节一致。未用有本地改动的工作树代替 commit 内容。
3. 本地 upstream 的 remote 名为 `vllm-project/vllm-ascend`，旧 NOTICE 使用 `Kaitounb/vllm-ascend` 名称；两份原元数据保留。这里以已核实 commit／路径／内容为 pin，不据此推断该 commit 当前可从哪个公共仓库抓取。公共固定链接本次未能获取，不标在线复现成功。
4. 从旧工程复制 `COPYING.vllm-ascend` 与 `NOTICE.md` 到 `provenance/licenses/`。COPYING 与本地 upstream 的 `git show HEAD:LICENSE` hash 一致；许可文本适用于上述 GDN 来源，不赋予新项目统一许可证。references 快照原文件未动。
5. 其余 learning 文件的来源与 hash 可锁定，缺少明确许可／revision 的字段如实记录为 unknown；本次仍限定为 reference_only，后续改编前补齐，不自动“视为 MIT/Apache”。本任务的验收是审查并确定采用边界，不表示所有参考材料的再分发许可已解决。

完整机器记录见 [reference-lock.json](reference-lock.json)，许可归档说明见 [README.md](README.md)。后续新增改编或升级 reference 要新建版本记录，不能覆盖现有原始证据来让锁定检查通过。

# 小规模36提案结果（P3-04）

修正版小规模流程已完成2锚点×A/B/C×1次搜索×6提案，共36attempts；B/C实际调用24次，106181tokens，零晋升。它验证了预算/状态/工具/缓存/拒绝闭环，不是正式P6消融，也不能据此宣称NCU改善或恶化了搜索质量。

| 任务/方法 | 提案 | 拒绝/原配置重复 | 新API/token | 最终 |
|---|---:|---:|---:|---|
| GEMM A | 6 | 5/1 | 0/0 | 原4-warps搜索参照 |
| GEMM B | 6 | 6/0 | 6/15975 | 原4-warps搜索参照 |
| GEMM C | 6 | 6/0 | 6/21775 | 原4-warps搜索参照 |
| MLA A | 6 | 4/2 | 0/0 | page16/split1 |
| MLA B | 6 | 0/6 | 6/31132 | page16/split1 |
| MLA C | 6 | 0/6 | 6/37299 | page16/split1 |

GEMM生产运行默认仍为cuBLAS，4-warps只是优化参照。每个attempt，包括无效/原样重复/静态重复/编译前拒绝，都消耗提案预算。精确已有身份复用已审计拒绝，不重新采样。全局700次账本防止失败或未知网络调用重试；24次request SHA、response原文、usage、方法自身历史及每次终态逐项审计通过。B从未调用profile loader或收到profile字段，C获得原生指标和明确成本。

新物理证据：warp1可信控制通过前后28项QA与3进程A/A但最坏2.609%拒绝；MLA page1/page64分别最坏3.606%/4.331%拒绝。GEMM B/C相同的新地址局部变量提取源码 `239802...` 只运行一次：独立builder59.407秒，320项host-FP64校验、两Sanitizer各8项0errors，3独立A/A共864段最坏2.172%拒绝。另一次仅局部变量改名静态拒绝；未把语义等同当作新二进制性能结果。GEMM warp2/8与旧失败只引用原证据，不重采。

成本口径：各分支API wall约14.032/12.882/8.333/9.787秒（GEMM B/C、MLA B/C）；C初始profile分别3.724928秒和4.118022秒，CSV/真实binary绑定保留。上述新源码全部worker wall合计211.871秒，含builder。缓存命中边际GPU成本为0，原采集成本另外列出，不能用它证明公平的端到端成本优势。账单货币/完整总墙钟不可得时留null，不填0。1.0只是未晋升时的名义回退分数，不是新测得的精确加速比。

前一批4个真实提案因Git header提示合同不完整而提前停止，2个GEMM在GPU前拒绝、2个MLA返回原配置；4次仍计入全局额度，独立保留，不并入修正版36条。新提示显式列出diff --git/---/+++；hunk计数只用原Git --recount恢复，原失败仍留档。

小规模暴露的正式实验前缺口：当前执行反馈只有编译/正确性/延迟，未显式给最终筛选理由、重复状态和既往源码变换，模型多次重复/改名/返回no-op。正式协议应先补这些安全字段、明确优化目标和父版本延迟，再冻结三次独立搜索与停止条件。不得事后把本批称为更丰富反馈下的实验。

证据：`evidence/20261123-mini-trial-v2/{protocol,verification}.json`（36条请求/响应/裁决与来源）；原早停批`evidence/20261123-mini-trial/early-stop.json`；新源码原始498文件在`evidence/20261123-mini-gemm-pointer-source/`。CPU审计代码为`experiments/full_delivery/audit_mini_trial.py`，其schema审查失败与修复记录保留；未重跑GPU或API。MLA轨迹见`docs/MLA_OPTIMIZATION_CASE.md`。

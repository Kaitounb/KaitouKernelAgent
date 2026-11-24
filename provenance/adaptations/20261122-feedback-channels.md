# P3-03 B/C 信息边界

实施前来源：KernelAgent `e0647170da36ef9b059ac0bd3d60103aa4ed378b` 的 `kernel_perf_agent/kernel_opt/profiler/ncu_profiler.py::load_ncu_metrics/metrics_to_prompt`、metric_schema 及已验收 `experiments/p1_04/ncu_adapter.py`；provider 继续采用已安装 bundle 内原 OpenAI-compatible provider。KDA `acdd79343c5e9d0c7cb96289f26fd3f1051ce785` 的 evidence/validation/promotion 分层沿用既有 source-binding/decision 适配，不重写控制框架。

新增缺口：旧硬件反馈实验只有 C 通道，缺少实际阻止 B 读取 profile 的构造入口。新增一个可离线运行的提示构造器，A/B 不调用 profile loader，不读取 profile 路径；C 仅接受带来源hash的结构化事实，全部输入显式白名单。编译反馈使用错误类别，不把混有 ptxas 资源统计的原始stdout/stderr给B。共享任务/模型/修改空间完全相同，仅 feedback.profile 有差异；模型调用、编译和评测不进入此构造器。

验证：CPU poison-loader、递归多余字段、同一提示主体、NaN/伪造hash/未知metric、真实历史profile构造与成本留档检查。无新API、无GPU、无B/C收益结论。测试通过仅代表信息构造边界，不替代P3-04真实搜索。

总调用预算缺口适配：既有command只有单次reservation，完整实验需跨run总上限。新增CallBudget沿用同样flock/fsync/atomic replace模式，授权文件SHA固定，reservation在API前落盘；未知/失败不返还额度，重复key从不重新授权网络调用。3项CPU故障/幂等测试通过。新预算700次，历史3次另列；不改变旧单次证据。

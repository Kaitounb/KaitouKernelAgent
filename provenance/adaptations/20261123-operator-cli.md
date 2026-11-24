# Operator 可安装入口接入前记录

复用 FlashInfer 0.2.8 `JitSpec.build_and_load/aot_path` 与既有 `OperatorRunner`；控制侧文件读取、fsync/atomic rename、独立worker失败后恢复规则沿用项目已验收的 `registry.py::{read_regular,atomic_json}`、`deployment.py::execute`。旧GEMM bundle45、RC1 wheel、默认cuBLAS和九项保护文件不改。

必要新增：独立 `kernel-agent-operators` 命令（不会给模型源码执行权限），只接受内置catalog的已验收SM86 bundle ID/manifest SHA；普通推理不带故障字段，故障注入只供显式恢复演练。输入先按冻结shape/dtype/字节预算读取，GPU worker不挂参考或凭据。已有native+QA证据可按逐字hash继承；Graph索引预分配改变的host适配器做安装后MLA定向回归。新wheel版本dev6，旧dev5文件保留供回滚，不用同版本替换wheel。

验证：新prefix/空cwd的`kernel-agent-operators inspect/run`，Softmax/Attention/MLA独立参考比对；crash/timeout后fresh worker恢复；损坏bundle拒绝（不执行GPU）；显式重新安装旧wheel及原GEMM status/run验证版本回退。仅工程开发部署，不提前宣称最终生产发布或收益。

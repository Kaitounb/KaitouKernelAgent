# 部署稳定性标准 v2：单对界限 3%

2026-11-23，用户明确提出接受3%单对A/A噪声。新增workloads/gemm_deployment_stability_v2.json；完整三session、8开发workload、1/16输入、9对交替顺序不变，单对偏差<3%，组几何均值偏离1仍<1.5%。原研究/搜索收益门槛、正确性、.97非退化保护项及旧证据不改。

这次标准修订发生在观察结果之后。以下是按新部署政策对历史数据的CPU复核，不能写成预先冻结的新独立验证，也不追改原2%协议的拒绝。

| 原批次 | 三轮是否完整 | 最大单对偏差 | 最大组均值偏差 | v2 A/A单项 |
|---|---|---|---|---|
| 原始cuBLAS | 是 | 4.274482% | 0.788516% | 不通过 |
| 锁频250ms | 是 | 2.849277% | 0.496481% | 通过 |
| 锁频1秒/HAGS Off | 是 | 2.040408% | 0.286466% | 通过 |
| 重启后HAGS On | 否，仅AA0 | 5.142371% | 1.170639% | 不通过 |

逐文件SHA归档与原始report重新复算；四批全部展示，无新增GPU采样或模型调用。最接近的一批有三轮864原始段，新标准下A/A单项通过，但没有A/B、没有证书或生产发布。该批为HAGS Off/锁频/1秒持续平均环境；不覆盖当前HAGS On环境，不保证Windows日常同时用卡时的延迟。

CPU结果：[verification.json](../evidence/20261123-deployment-stability-v2/verification.json)。旧2%审计、protocol、wheel与冻结研究文件保持，未把旧拒绝覆盖为通过。

## 接下来的4090验证

优先使用能预留GPU空闲时段的Linux实例。NCU是可选条件；可信cuBLAS的CUDA Event A/A/A/B不读取硬件性能计数器。Docker GPU/独立低权限执行能力有利于直接推进完整Agent部署，依[NVIDIA Container Toolkit官方配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)实际预检，不以“容器root”或有SSH便认定隔离可用。

没有Docker也可先验证已审查固定cuBLAS/SDK的计时；此时不执行自由生成源码，也不能据此声称完整源码worker部署验收。现4090以前的MLA A/A曾三session通过、最坏约0.307%，只说明该算子/旧环境的事实，不预先保证新GEMM会达标。

收到连接后的顺序：只读核验卡型/驱动/CUDA/Torch与GPU占用、持久目录和隔离条件；按实际SM89环境构建可信cuBLAS并做公开正确性/恢复检查；绑定新产物/环境/协议与本v2政策SHA；完整三轮A/A通过后执行三轮A/B和非退化保护项；CPU原始证据审计后再决定该目标卡版本的生产资格。不重跑3070封存独立清单，不把4090通过回填成3070通过。

旧服务器和本机归档保留，新连接先存私有next文件，不覆写旧凭据；未收到连接前不发起GPU工作或新增租赁/续费。

## 新环境的实际确认

4090已收到连接并按本v2在任何Event之前冻结：三AA+三AB共1728原始段、320QA、32Sanitizer及两fault/fresh恢复均通过；AA最大单对0.664015%、组偏差0.570454%，AB最低0.999909889×，917文件CPU审计PASS。因此本次存在真正的新SM89环境预先冻结确认；前面的3070历史复核仍保持“观察后修订”标签。可信native范围、软件版本、数据与成本见[SM89报告](SM89_CUBLAS_STABILITY.md)。

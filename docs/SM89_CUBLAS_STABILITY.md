# 4090可信cuBLAS稳定性验证通过

2026-11-23，用户提供可用4090后，在该实例实测并独立CPU审计：三A/A与三A/B全部完成，8公开开发workload×1/16输入策略，共1728原始CUDA Event段；v2部署稳定性和非退化保护通过。

| 验收项 | 实际结果 |
|---|---|
| 公开正确性 | 320项通过；CPU FP64参考在可信主机独立计算，账本最大绝对误差2.28881836e-05 |
| Sanitizer | 两绑定分别memcheck/synccheck，共32项检查、0 errors |
| 故障恢复 | crash/有限CUDA后host timeout均拒绝，两个fresh进程恢复通过 |
| A/A | 三独立worker session、864段；最坏单对0.664015%，最坏组均值偏差0.570454% |
| A/B非退化 | 三独立session、864段，16项workload/策略汇总均≥0.97，最低0.999909889× |
| 原始证据 | 917文件逐SHA、NKA1/NPY、保护文件、契约见证、Graph输出/输入哈希、pilot与原门槛复算通过 |
| 成本 | 真实Event累计1667.240s；0新DeepSeek、0NCU、0封存验收 |

## 三轮A/A

| Session | 最大单对偏差 | 最大组均值偏差 | v2结果 |
|---|---|---|---|
| 0 | 0.594498% | 0.393596% | 通过 |
| 1 | 0.664015% | 0.570454% | 通过 |
| 2 | 0.476449% | 0.421494% | 通过 |

新v2单对<3%、组均值偏差<1.5%在任何本轮Event之前冻结；实际三轮也满足旧2%单对界限。旧SM86的2%拒绝及用户修订3%的回顾复核均保留，不以本轮覆盖3070结果。

## 运行范围与产物

GPU：RTX4090/SM89，UUID GPU-a1a52305-25e4-75a4-cbb8-0d6d8e934138，driver570.86.10，Ubuntu22.04容器/内核5.15，Python3.12.12、Torch2.7.1+cu126、CUDA12.6.85。原项目venv仍在；NVCC/ninja已有，只适配实际PATH，没有重新安装系统软件。

cuBLAS CUDA代码与合同原字节复用；两个新SM89绑定仅角色文档字符串不同，使用固定KernelBench loader原函数及load_inline构建。incumbent SHA `4572c346c20a70630a2521eac8c4a83163f26d931a7d07bbcd549e28220a43de`；candidate SHA `72d466d30e6b777b06d8e4216bdd3f4aef1306127926ac6f0976923d3406009d`。编译源码、完整命令/build.ninja、函数及模块独立身份均留档。KernelBench许可随原dev6资产完整保留，本轮sources/upstream另存许可。

计时继续使用原Graph/Event路径：32调用/Graph、3×4replay pilot取median、replays=clamp(ceil(4000/pilot_ms),1,8192)、每组9对交替顺序、目标1秒持续平均。动态Linux时钟未手动调整；时钟/温度/显存原始采样保留。采样内高负载记录3047条，SM [2355.0, 2670.0]MHz、显存[10251.0, 10251.0]MHz，最高69.0℃，全阶段峰值显存1471.0MiB。这些是环境描述，不是单请求尾延迟保证。

本轮是同一可信cuBLAS的独立绑定与计时回归；原5%收益门槛没有改变，原研究裁决仍为REJECT/insufficient_gain_or_regression，无优化提速或best晋升声明。

实例无Docker，本轮只执行已审查固定cuBLAS/SDK，native子进程复用原字节/时间边界和原始帧；完整模型源码worker的隔离未验收。参考不放入worker请求，但native同UID不具备文件系统参考隔离。本次通过覆盖SM89可信路径的稳定性/非退化；生产发布还需目标环境安装、版本回滚和相应执行边界验收，production_release保持false。

## 证据与CPU复现

- [CPU审计](../evidence/20261123-sm89-cublas-v2/verification.json)、[完整摘要](../evidence/20261123-sm89-cublas-v2/summary.json)、[逐workload CSV](../evidence/20261123-sm89-cublas-v2/per-workload.csv)、[成本](../evidence/20261123-sm89-cublas-v2/costs.json)。
- protocol SHA `8a023ee57b57948c72866304e74dabd4f50fe27a2249278473c31e4572961065`；原始归档SHA `68eb546027baf486b5db2ff46bca89876f7fa6bcf42696d8e0bb99d89ea5a86b`。
- 只读CPU审计命令：`/tmp/nka-delivery-verify-venv/bin/python -m experiments.full_delivery.audit_sm89_cublas evidence/20261123-sm89-cublas-v2`。
- GPU入口已消耗该身份；保留原始数据，不重采。远端原目录`/data/nka-cublas-v2/sm89-f78360gq`保留。新连接在本机私有next文件，旧连接/备份不覆盖。验证后0%GPU、1MiB、无compute进程。

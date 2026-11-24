# 正式 A/B/C 实验结果

54组冻结搜索全部结束，实际221个提案、158次DeepSeek调用、858,008 tokens；112次重复、109次拒绝，**0次完整筛选通过**。保留初始实现，未激活任何模型候选。本批没有建立硬件反馈收益，也没有达到1.03/1.05的合格优化目标。

协议SHA256 `56b6bdbe1ba587c9dac3b84d59f81146e120767787db9c8529de129fdc698e7f`。GEMM dev-01/02/06、MLA mla-dev-02/04/08，每目标A/B/C各3次搜索。[冻结协议](FORMAL_STUDY_PROTOCOL.md)

## 搜索与裁决

| 方法 | 搜索组 | 实际提案 | 模型调用 | tokens | 重复 | 拒绝 | 筛选通过 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 参数搜索 | 18 | 63 | 0 | 0 | 18 | 45 | 0 |
| B 模型＋执行反馈 | 18 | 81 | 81 | 382,695 | 51 | 30 | 0 |
| C B＋初始父版本NCU | 18 | 77 | 77 | 475,313 | 43 | 34 | 0 |

18组A遍历有限空间后停止；36组B/C按连续3次重复/keep_parent/无效提案停止。每组最多12、全体最多648提案是上限，不要求凑满；实际最长搜索10个提案。模型、温度、输出长度、允许修改范围和提示主体保持冻结，B不读profile loader，C使用18份实测父版本诊断。所有调用无自动重试。

新增13个源码身份实际评测：1个在无GPU的builder内编译失败；其余12个各完成320项独立控制侧FP64正确性校验，以及memcheck/synccheck各8项、0 errors。合计3,840项QA、192项Sanitizer输出检查。7个身份A/A不稳定，5个通过A/A并完成A/B；共保留14,688个原始Event段。不同源码/DSO身份不代表不同CUDA机器码或独立算法。

5个进入A/B的身份，以下GM均按原工程anchors聚合，保留全部8个workload保护项：

| 身份前缀 | 单输入GM | 16输入ring GM | 最差保护项（两策略） | 裁决 |
|---|---:|---:|---:|---|
| 907f4f3f16d4 地址局部量 | 0.999528x | 1.000138x | 0.999170 / 0.999966 | 收益不足 |
| 1754b6006ecb 整数偏移局部量 | 0.999863x | 1.000694x | 0.999568 / 0.999635 | 收益不足 |
| 5165fcf73698 builtin assume | 0.999275x | 0.999265x | 0.998914 / 0.998958 | 收益不足 |
| 5f4d3b024fda 地址外提＋输出指针 | 1.000347x | 0.999873x | 0.668179 / 0.617947 | 保护项退化 |
| 20408b6ac35b shared-B＋地址外提 | 0.959699x | 0.949130x | 0.334207 / 0.321008 | 退化 |

原GM≥1.05、区间下界>1.02、每个保护项≥0.97和两输入策略规则未放宽。旧A/A拒绝身份未重采。纯局部变量改名通过完整源码逐token替换的字节相等证明静态拒绝，不继承其他源码数值结果。新增不合法WMMA逐lane指针、错误shared-A索引等在GPU前拒绝；原始响应、patch、严格Git失败和允许的hunk计数恢复均保留。

MLA的54次B/C提案均保留page16/split1，没有提出新的配置；不是穷尽了B/C允许的全部page/split空间。A的27个提案遍历其预定page1/16/64、split1有限空间并复用既有精确结果。此前真实split2双卡负结果、147配置正确性和跨卡普通调参另存，不能倒填为本批新样本。

## 成本与曲线

| 成本子阶段的累计墙钟秒 | A | B | C |
|---|---:|---:|---:|
| API请求 | 0 | 150.640 | 164.170 |
| 本组初始父版本profile | 0 | 0 | 78.061 |
| 新身份编译＋worker | 0 | 1,382.958 | 1,881.430 |

新身份编译累计849.085秒，已经包含在3,264.388秒worker成本内，不重复相加。原始Event段累计284.840秒；这是定时段的设备Event时间，不是全部GPU占用时间。API累计314.810秒、accepted初始profile累计78.061秒。

表中0表示本批没有该项新的实测工作。A及其他精确cache命中的历史构建/QA/计时成本另列于初始cache proof，不能称为零成本算法。失败的MLA JIT身份诊断、早期试验、归档/传输、CPU编排和端到端经过时间不混入上述子阶段。异步请求共享物理评测cache，首次请求承担新身份成本，因此该分配不能证明某方法具有因果上的成本优势。服务未返回实际账单金额，货币费用为null，不估算。

当前700次新增授权已用190次、997,897 tokens，余510次；历史3次/8,790 tokens单列，合计193次/1,006,687 tokens。全局190个调用均有RECEIVED及usage记录，无未知调用被填成0成本。

[提案与成本图](../evidence/20261123-formal-study-v1/report-final/study.png) · [token/子阶段墙钟曲线](../evidence/20261123-formal-study-v1/report-final/budget-curves.png)。曲线中的保留分数1.0仅表示默认未变，不是假造测得1.0x，也不代表达到优化目标。金额未知，token曲线不称为货币费用曲线；子阶段墙钟不称为完整端到端墙钟。

## 解释范围

- C收到固定初始父版本的native kernel诊断，没有逐候选更新profile；MLA诊断不包括wrapper/combine。不能把本批等同完整自适应硬件反馈搜索。
- 共同提示给出所选目标shape，没有列出其余7个保护shape。执行延迟反馈为首session、单输入策略的候选样本；完整guard、16-input策略及三session结果留在外置证据中。
- 安全反馈把收益不足和保护项退化合并为一个枚举，也没有详细编译/静态错误说明。B/C共同受到此限制，不能将保守/重复提案全部归因为NCU指标无价值。
- 独立的是模型搜索历史；相同身份复用物理GPU结果，cache命中不是新的独立性能样本。源码身份不同也不保证机器码不同。
- 大量身份受SM86 A/A噪声拒绝；其A/B没有运行，不能填收益或退化值。没有证据证明默认WMMA已接近硬件最优。
- 4090目标C受计数器权限限制，未运行。普通迁移/再调参见[跨卡结果](CROSSCARD_RESULTS.md)，不能冒称完整跨卡消融。

## 工程交付状态

正式实验完成不等于生产发布。默认cuBLAS额外部署资格的320项正确性通过，但三A/A最坏4.274%超门槛，未进入A/B、未生成资格证书。[生产阻塞](PRODUCTION_BLOCKERS.md)

`0.1.0rc1`预览新prefix/空cwd安装通过19项CPU、5份GPU输出、7次尝试和实际包回滚；qualified通道缺证书时GPU前拒绝。[预览包与安装命令](RC1_PREVIEW.md)。MLA默认独立4项×2seed在用户具体批准后按原冻结协议一次执行，8/8调用、16/16输出及204私有文件CPU复核通过；精确值未显示给模型（[独立验收](MLA_RC1_ACCEPTANCE.md)）。生产标志保持false。

## 原始证据与复算

主目录 `evidence/20261123-formal-study-v1` 包含protocol/freeze、每个request/response/usage、原始patch、规范化eval-input、review、decision、closed、唯一评测reservation和精确cache。`verification.json`逐条重建自身历史、B/C边界、实际响应→patch/config→descriptor，并验证所有证据SHA。真实GPU输出与外部FP64数值判定归于对应可信host evaluator；本机CPU审计复核帧、NPY、receipt、边界和原始判定，不冒称新GPU运行。

`report-final`保留逐提案CSV、54组搜索CSV、13个物理身份成本、1,632条逐workload/session/policy/role延迟行、搜索曲线、PNG/SVG及manifest。GPU数据在各 `evidence/20261123-formal-gemm-*` 目录，SSH认证失败B219另有未执行证明与transport-correction；原始失败未删除。

```sh
python -m experiments.full_delivery.audit_formal_study evidence/20261123-formal-study-v1
python -m experiments.full_delivery.audit_cublas_qualification evidence/20261123-cublas-final-qualification
python -m experiments.full_delivery.audit_rc1_preview evidence/20261123-rc1-preview-installed
python tools/verify_gemm_release_freeze.py
```

以上均为离线复核，不新增API/GPU采样或读取封存清单。绘图使用已记录的NumPy2.5.5/Matplotlib3.11.2；需要重绘时指定新的输出目录，保留已有文件。

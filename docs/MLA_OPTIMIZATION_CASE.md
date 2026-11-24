# MLA 完整拒绝案例：SDK 已内部 split 时再显式 split

当前保留固定 FlashInfer0.2.8 fa2 的 page16/显式split1，未晋升新配置，也未获得稳定收益。本文记录核心 Z/自然LSE 的优化轨迹；不代表完整 DeepSeek 层或模型吞吐。

起点为8项冻结开发workload，R512/Drope64、FP16量化输入、CPU FP64独立参考、原始scale=1/sqrt192。SDK自带内部split planner。工程适配包含随机分页、空请求、自然/二进制LSE转换、H128 merge按32个独立heads分组及Graph预建索引；不修改原CUDA算法。两卡各147配置及关键memcheck/synccheck已通过。

3070原生NCU的3launch/19字段绑定实际DSO `cdc4c3ba11ef61b7e2778f93af247bc0ac53027daa8035c55b541e6ea3fd601e`，采集4.118秒。长KV样例显示252寄存器/线程、93696字节shared/block、16.66% active warps及34.95% barrier stalls。它们是实测事实，不能单独证明修改后的收益。两路DeepSeek在相同起点各提出page16/split2；B只看执行，C另看上述事实。两者源码配置完全重复，因此只评测一次、分别保留请求/usage。

| 卡/配置 | 三会话自身A/A最坏单对 | A/B锚点GM（1输入/16轮转） | 结果 |
|---|---:|---:|---|
| 3070 page16/split1基线校准 | 3.184% | 未测 | 校准拒绝，不重采 |
| 3070 page16/split2 | 1.257% | 0.56851/0.57309 | 明显退化，拒绝 |
| 3070 page1/split1控制 | 3.606% | A/A失败，跳过 | 拒绝，不重采 |
| 3070 page64/split1控制 | 4.331% | A/A失败，跳过 | 拒绝，不重采 |
| 4090 page16/split1基线校准 | 0.307% | 未测 | 校准通过 |
| 4090 page16/split2直接迁移 | 0.307% | 0.49158/0.49441 | 明显退化，拒绝 |
| 4090 page64/split1普通再调参 | 0.918% | 1.00097/1.00122 | 不足1.05门槛，拒绝 |

split2的两卡每张保留1728段，page1/page64在3070分别保留864段、4090 page64保留1728段；全部可由各目录verification.json独立复算。每个新配置均先做CPU参考/输入hash/offset guard/非默认stream校验，再三独立进程A/A，通过才三A/B；完整partial+combine都计入Event。每Graph32次core、每段32replay、9交替配对，1秒热身；metadata/布局/H2D/JIT在Event外。16套输入只是轮转，不保证cold cache。峰值约939MB，低于冻结2.217GB保守预算。

失败假设已明确：SDK已经内部拆分，外部再split未降低整体路径成本，增加了partial调用与merge及相关张量操作。实际Event证实方案变慢，计数器不能替代这项测量。page64在4090也只有约0.1%差异，不满足冻结GM≥1.05、区间下界>1.02、每项≥0.97的筛选规则。3070基线噪声未过，split2自身A/A通过也不能倒填基线校准为通过；该卡不作最终稳定收益结论。

首次Graph capture因现场CPU→CUDA索引拷贝失败、SDK H128 merge超线程失败、最初编译身份漂移和本任务GPU重叠批均完整保留。后两批未用于结论。修复采用预建固定索引及同卡flock；未放宽数值/噪声门槛。4090 NCU实际ERR_NVGPUCTRPERM，无虚构指标。

原始轨迹与审计：

- `evidence/20261123-mla-live-config/verification.json`：同起点真实B/C、2次调用11749tokens及两卡关联。
- `evidence/20261123-mla-live-split2-{sm86,sm89}/`：QA、实际DSO、3AA+3AB、GPU监测、所有Event。
- `evidence/20261123-mini-mla-page{1,64}-sm86/`：普通参数控制的完整A/A拒绝。
- `evidence/20261123-crosscard-mla-page64-sm89/`：目标卡普通再调参的完整拒绝。
- `evidence/20261123-mla-ncu-sm86-pinned-native/`：CSV/.ncu-rep/真实native库与采集成本。

该轨迹满足成熟CUDA配置/显式split图的优化与失败回退案例，不是任意MLA CUDA源码重写、正式A/B/C消融或生产发布。独立4项MLA shape/seed尚未读取。

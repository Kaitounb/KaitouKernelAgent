# 正式实验冻结协议

协议：`evidence/20261123-formal-study-v1/protocol.json`，SHA256 `56b6bdbe1ba587c9dac3b84d59f81146e120767787db9c8529de129fdc698e7f`。在正式模型调用前冻结；更早草案及调用前的停止规则修正完整保存在 `evidence/20261123-formal-study-initial-freeze-before-dispatch`。独立验收值不进入任务、反馈或模型请求。

6 个公开搜索目标：GEMM dev-01/02/06，MLA mla-dev-02/04/08；A/B/C 各 3 次独立搜索，共 54 组。每组最多 12 个提案，首次全局筛选通过、连续 3 次重复或无效提案时停止；A 遍历有限配置后停止。648 个提案、432 次模型调用都是上限。失败、重复、keep_parent 和未知调用均计数，不为凑满次数重采 GPU。

| 方法 | 输入和变换范围 | 额外成本 |
|---|---|---|
| A | 固定实现的合法配置；GEMM warps=1/2/4/8，MLA page=1/16/64、split=1 | 确定性搜索；没有模型调用 |
| B | 源码、合同、固定初始父延迟、本方法的编译/正确性/计时裁决、既往 patch/config | 实际模型 token 和请求墙钟 |
| C | B 的相同输入，另给与父源码/二进制绑定的初始 NCU profile | 每组单独的实际 profile 墙钟 |

B/C 同为 DeepSeek `deepseek-flash`、temperature=0.2、max_tokens=768、单次 60 秒、无自动重试。服务未提供可复现采样 seed，保留所有原始响应。每次 patch 都作用于冻结的初始源码，只传本方法自己的历史。GEMM 保留 FP32 累加顺序和 host dispatcher；MLA 仅允许原 FlashInfer CUDA SDK 的 page/split 图配置。A 是实用参照；B/C 是硬件信息的对照。全局另受用户批准的 700 次新调用硬上限约束。

C 的信息是**固定初始父版本诊断**，不声称逐候选动态采集。18 份合格 profile 共 54 次 native launch、每份 19 字段，实际总墙钟 78.060903 秒。SM86 MLA 严格使用已测 fa2 `cdc4c3…` / cascade `31be5d…` 的原 JitSpec AOT 布局。首次 9 份 MLA 诊断发生 JIT 二进制身份漂移，已拒绝并完整留档；只有受影响的 9 份重新采集。实际 Ninja 仍为 sm86，不归因为 PTX。profile 不替代 Event，MLA native 指标不覆盖 Python wrapper 或 combine。

每份新源码先经可信审查，再进入固定 dev5 的低权限 Docker builder/worker。UID65534、无外网、只读挂载、无新增 capabilities；参考由容器外计算。唯一身份先预留，独占 GPU 锁串行执行。GPU 正确性、Sanitizer、三 A/A 后才允许三 A/B。MLA 新配置必须使用冻结 SDK 字节。已有失败身份只复用原始结果，不能换个搜索分支重采；只有变量名变化的源码保留完整逐 token 对照，静态拒绝，不继承参考变体的数值结果。

工程筛选仍保护所有 8 个开发 workload、两种输入缓存策略；三 session，GM≥1.05、区间下界>1.02、保护项≥0.97。选择可修改的 GEMM dev-01/02/06 作为搜索目标不改变旧工程筛选 anchors 01/03/08。A/A 失败不能证明收益。通过筛选也不直接激活运行时，仍需独立验收与发布裁决。

成本分为实际 API、profile、编译/worker 子阶段，以及历史 cache 构建。命中 cache 的边际 GPU 工作为零，不把原始工作写成零成本。没有账单货币数据时金额为 null。没有通过筛选时，曲线只画“默认保留分数 1.0”，不声称测得 1.0x 加速或达到 1.03/1.05 目标。

CPU 独立复算：

```sh
python -m experiments.full_delivery.audit_formal_study evidence/20261123-formal-study-v1
```

实验未结束时可加 `--allow-incomplete`，输出明确为 INCOMPLETE。冻结之后新增的结果接纳和审计模块不修改协议、阈值、提示或采样工具。4090 的 NCU 仍因平台计数器权限不可用；该限制必须在跨卡报告中单列。

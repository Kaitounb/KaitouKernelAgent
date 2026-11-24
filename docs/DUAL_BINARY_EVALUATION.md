# 双二进制 A/B 与外置开发筛选

2026-11-22，GEMM-REL-02完成开发组件验收：已安装的低权限worker能分别加载两个同名Torch扩展，独立验证两侧输出，保留原配对Event路径，并将source／binary／harness／protocol绑定到外置筛选。没有真实新候选晋升或生产发布。

最终开发wheel：[0.1.0.dev5](../evidence/20261122-gemm-rel02-dual-installed-fixed-test-import/remote/run/dist/nvidia_kernel_agent-0.1.0.dev5-py3-none-any.whl)，SHA256 `348a6e6ed8fca9051b570fb96548d4491c59a14c8f261df2b3e0e2172e5d2c7a`；45份优化资源／模块的bundle manifest SHA256 `46dc2fd0f6997c5c96e6a89075a686531a1e65587e93f0db8762d4a1bad6e8f5`。原dev4等wheel与证据保留。[安装命令和单次预算](OFFLINE_OPTIMIZER.md)仍适用，替换wheel版本即可。

## 新候选路径

`optimize`普通模式已接到双binary pipeline；`--checks-only`保留原单候选正确性入口，不允许产生KEEP。

1. 候选经guard／控制侧源码审查绑定，无GPU builder从固定KernelBench／Torch loader fresh build。parent固定为当前4-warps搜索参照，incumbent实际binary为已验证的93fb…产物；候选binary、三CUDA source hash、五项合同／identity、固定协议、45项bundle与6份mounted worker hash进入descriptor。
2. 使用CPython原`importlib.util.spec_from_file_location`，模块全名分别为`_nka_incumbent.nka_gemm_v1`和`_nka_candidate.nka_gemm_v1`。保留末级PyInit名字，不改ABI、不写FFI。两份.so都在同一artifact只读挂载中；总计仍只有artifact／inputs两份宿主只读挂载。
3. 两侧各160项输出由可信宿主预先生成的CPU FP64参考校验，分别建立私有ledger；ledger和参考答案不发child。candidate在双加载上下文中完成memcheck／synccheck。输出、输入bit hash、offset guard、producer/candidate/consumer stream及两侧module/function/path身份都绑定。
4. 新合格source在GPU连续3次util0后只执行预注册三session candidate A/A。此时parent也已加载，避免与后续A/B使用不同DSO上下文。不稳、缺失或cleanup未确认立即停止；known failure和重复identity不启动采样。
5. A/A全通过且GPU再次空闲后，才启动三session双binary A/B。沿用原32图节点／12replay／9对、1秒warmup、交替捕获／Event顺序，计时函数体AST未变。两侧各32项合同在每session测前后核对，图中32次输出也必须匹配各自FP64验收过的ledger。
6. 原冻结统计不改：两缓存策略分别满足锚点GM≥1.05、探索区间下界>1.02、所有开发保护项≥0.97；A/A单对偏差<2%、group GM偏离<1.5%。逐session log、4.303区间和分母与原gate CPU对照一致。

`pair_decision.decide_pair`只输出开发筛选建议。KEEP落盘为`SCREENING_PASS`及`decision.json`，不写已冻结best、不改运行时allowlist／发布指针；默认仍cuBLAS。候选正式冻结、独立验收、专家对照／部署选择另行裁定。明确不宣称任意恶意native计时防伪或驱动隔离。

## 本轮实际验证

本轮没有新模型提案，真实模型调用仍累计三次8,790 tokens；没有对旧P1-05或已拒绝P3-01候选重采A/A。完整新候选320项QA→三A/A→三A/B路径已实现，但本轮未用新source执行该完整收益协议，不将模拟KEEP当实卡晋升。

| 验证 | 实际结果 |
|---|---|
| 新prefix、空cwd安装 | dev5安装／资源校验通过；16项优化入口＋19项双binary gate／调度CPU测试通过 |
| 同名扩展分离 | 正确93fb／66fcc两个artifact、93fb／已审查零输出a4a25…故障组合；两角色各4项，共16项。12正确、故障candidate4项全部数值拒绝，baseline不受污染 |
| 双加载Sanitizer | memcheck／synccheck各4项，均明确0errors |
| artifact hash不符 | 两侧各一次在创建GPU job前拒绝 |
| 独立诊断输出 | WMMA／cuBLAS两算法、两shape×16seeds×两角色64项FP64输出，另4项cuBLAS合同输出；全轮共92项输出校验，含上述4个预期失败 |
| 诊断A/B通道 | GPU连续三次空闲后，一session、两shape×两缓存×9对×2段=72原始Event段；测前后16项合同witness及256个图输出hash通过 |
| 外置CPU裁决 | 正例仅CPU_SIMULATION；无收益／高不确定性／保护项退化／两角色输出混淆／缺失／篡改／未确认清理均拒绝。busy、A/A不稳／invalid时不启动A/B，known rejection不重新采样 |

诊断shape固定为128³和32×48×48，明确是93fb WMMA4对66fcc cuBLAS，不是把两个相同WMMA换名后重跑A/A。report.backends明确`template/cublas`；production source gate只接受`template/template`，并要求8开发项、三session自身A/A等完整绑定，因此这72段不可用于新source的KEEP。所有原始段保留，不追加挑选重测，不报告优化收益。

首次安装验证只在CPU原gate对照测试中因仓库import失败，尚未启动GPU；原日志／wheel留档。修正测试从包内原gate快照导入后，重新安装同一dev5组件并通过全部测试，未修改已完成的GPU批次。

## 证据、复算及后续

[实卡／完整原始证据](../evidence/20261122-gemm-rel02-dual-installed-fixed-test-import/) · [独立CPU审计](../evidence/20261122-gemm-rel02-dual-installed-fixed-test-import/verification.json) · [实施前来源／协议](../provenance/adaptations/20261122-gemm-rel-02-dual-binary.md)。审计重解析wheel／manifest／NKA1／标准NPY／role witness／cleanup／全部72 Event段；数值验收来自独立可信3070宿主，不宣称本机重新做了GPU／FP64实验。

```sh
python3 -m experiments.gemm_rel02.audit --evidence evidence/20261122-gemm-rel02-dual-installed-fixed-test-import --output /tmp/new-dual-audit.json
python3 -m unittest discover -s tests -p 'test_pair_decision.py' -v
python3 -m unittest discover -s tests -p 'test_optimizer.py' -v
```

下一项GEMM-REL-03：冻结首批GEMM发布候选和运行支持范围，先固定独立验收协议、产物选择和停止／回滚规则，再安排封存验收。本轮不读取／执行封存清单，不重置API预算。P1-05计时未验收、P3-01失败原样保留；独立发布及正式性能结论仍待验收。

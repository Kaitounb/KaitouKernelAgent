# GEMM v1 冻结合同与 workload

`gemm_contract_v1.json` 固定 SM86、连续行优先 FP16 输入／FP32 累加与输出，正维度不超过 4096。输入域、容差、非法输入行为、stream、dispatch 和计时边界均在合同中声明。参考由候选执行前的相同 FP16 量化值在 CPU FP64 计算，容差沿用已验证接入的 `atol=3e-5, rtol=1e-4`。

`gemm_development_v1.json` 冻结 8 个开发 workload，含方阵、小 M、矩形和非整齐尺寸。随机输入使用 seed 171/172；全零、带符号大有限值、极小形状、尾部和非零 storage offset 属于额外正确性边界，不计入性能分母。

4 个独立验收 workload 从未与开发集重复的 shape 家族成员中一次生成，精确 shape 与 seeds 保存在 `.private/gemm_acceptance_v1.json`；公开 commitment 文件只记录 SHA256、数量与 family。当前仅冻结，**尚未运行独立验收**。该文件不传入候选构建／运行目录，不提供给模型。候选冻结后由可信 evaluator 验证 commitment 并执行一次；若据此调参，该批转为开发集，重新建立独立验收集。

本地文件权限为 0600，但同 UID/root 仍可读。完整 evaluator/候选 OS 权限隔离待 P3-02，不能将文件路径和 commitment 宣称为完整隔离。生产候选侧不得挂载该清单或参考。

可信模板仅搜索 `warps ∈ {1,2,4,8}`，实际改变每个 block 的 warp 数／M tile 覆盖。整 tile 且满足 WMMA 对齐条件时运行 CUDA；其他合法输入调用 cuBLAS。运行路径由真实输入规格决定。当前尚无模型搜索、候选晋升或性能收益结论。

搜索开始前的计时／晋升规则冻结在 `gemm_measurement_v1.json`：三个锚点、两种缓存策略、每组 9 对、至少三独立进程；暂定锚点几何均值 ≥1.05、探索性区间下界 >1.02、各开发保护 workload ≥0.97，所有正确性通过。门槛高于 PLAN 的初始建议，是因为双 Graph cuBLAS A/A 实测存在约 1% 固定偏差；原始偏差保留，不能解释为收益。新候选若超出已校准延迟范围或改变计时模式，先单独 A/A。

Graph replay 的 Event 时间覆盖已捕获 GPU 工作，host 输出分配、Python 分派与 capture 不计入该时间。eager 调用延迟若后续需要，另行报告；不能将当前 Graph 时间称为完整部署端到端延迟。合同的 timing 字段在正确性批次后作此文字澄清，数值／shape／seed／容差没有变化；前后 hash 在 measurement 文件中可追溯。

# GEMM 首批候选与独立验收协议冻结（实施前）

2026-11-22。用户现明确要求连续做完整项目，阶段完成后不停止等“继续”。沿用原KDA/ForgeKernel的候选冻结→独立验证→发布分离流程；本阶段先冻结已实际安装的dev5包与cuBLAS默认／fallback两份artifact，再由独立可信evaluator按既有commitment执行一次。agent不读取精确独立shape／seed，候选不挂载清单／参考，模型不接收数据。

- KDA acdd79343c5e9d0c7cb96289f26fd3f1051ce785：source／binary／contract／protocol／支持范围和证据完整绑定，未通过不得发布。
- ForgeKernel b2c8ae8510e02c28576808910a6b8827721ab7dc：冻结后不调参，保留全部失败，不选择性重跑独立数据。
- KernelBench 423217d9fda91e0c2d67e4a43bf62f96f6d104f1：可信host FP64、原量化／容差／guard；固定CUDA runtime使用已验收绑定。
- KernelAgent e0647170da36ef9b059ac0bd3d60103aa4ed378b：本阶段0新模型调用；密钥只留本机控制侧。
- FlashInfer-Bench 15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee生命周期原则与已验收P3-02 DockerRunner：两份只读bind、非root、无网络、有界输出／外置timeout／cleanup；参考与清单留可信host。

部署选择只允许已验证cuBLAS66fcc／fallback93fb；source优化器KEEP不改变部署。独立验收是正确性／输入保护／安装／回退范围，不把已知不稳WMMA成绩或新诊断72段转为收益。正式性能结论另遵守原三会话协议；旧P1-05和P3-01 A/A失败不重采。

候选锁定后执行器先比对原公开commitment，创建私有一次性执行记录，实际运行后不隐式再执行。清单只由私有控制程序读，既不输出精确shape／seed，也不进入公开transport脚本；所有原始runtime记录／NPY等存在私有证据目录，公开仅汇总数量／误差／hash及scope。环境／transport在执行前失败可另目录诊断，不能抹除已有GPU失败。

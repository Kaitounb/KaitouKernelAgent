# 首批 GEMM 候选冻结及独立验收协议

2026-11-22。用户要求连续完成完整项目；首批候选先冻结为`gemm-sm86-rc1`。这是离线GEMM运行包的候选，不把开发优化器的CPU模拟KEEP、72段诊断或失败A/A改写为收益。

候选锁见`releases/gemm-sm86-rc1/freeze.json`，freeze ID `1756bace31e202585de4164d3f4dd68d53eee5da22c8f364e06334495f615549`；独立协议见同目录`protocol.json`。冻结dev5 wheel、默认cuBLAS66fcc／fallback93fb、数值合同、统计协议、45项优化资源和既有证据hash。支持范围仅为已实测RTX3070/SM86、指定固定镜像和Torch2.14+cu130／CUDA13.0，准确环境在freeze中；4090另建支持证据，不借用此资格。

公开独立清单commitment仍为原`b4c3a762…`，共4类workload。可信执行器只在冻结核对通过后读取精确清单，候选只收每次的实际输入NPY，不挂载清单、seed表、CPU答案、控制状态或模型凭据。agent不读取精确shape／seed。真实GPU运行后的所有原始NPY、参考判定、日志和错误留在私有证据目录；公开仅汇总通过数量、最大误差及证据commitment。

每个冻结seed、每个default／fallback artifact各执行一次；FP16量化后在可信host预先计算CPU FP64并转FP32，沿用atol3e-5／rtol1e-4。不得静默fallback遮掩某artifact失败，不选择性重采；cleanup未确认就停止后续GPU工作。运行启动预留持久化，传输失联不自动重试。若根据独立结果改kernel，该数据改标开发并重新建立独立验收。

本协议不采样性能，不重跑旧4-warps或7fd候选A/A。默认cuBLAS的可靠性与源码优化收益分别验收。公共17×23／23×19输入另做有限host crash／timeout fallback及版本回滚，不能拿它们替代独立集合。

冻结／协议本身不打开production flag，不激活新artifact；完整安装／正确性／恢复证据通过后再对声明范围裁定。完整项目中的Attention／MLA、跨卡与B/C试验继续独立推进，不因首批GEMM候选冻结宣称项目已经完成。

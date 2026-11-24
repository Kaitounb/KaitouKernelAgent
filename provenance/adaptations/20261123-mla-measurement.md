# MLA 配对计时接入前记录

上游：KernelBench `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` 的 CUDA Event/Graph 计时职责；项目已验收的 `kernel_agent/optimization/identity.py::group_log_ratio`、GEMM 双Graph控制器保留原18段/9对次序与三独立进程统计。FlashInfer `3f8317c67ac129facb8525b2fb581be42f634459` 的 MLA fa2 与 merge 原CUDA不改。具体运行仍调用现有 `SplitMLAPlan`，不另写kernel或loader。

新增缺口是MLA多输入、多kernel Graph及受控page/split配置的计时适配；GEMM workload/二进制/协议不能直接套用。新 `mla_paired.py` 每段32个Graph replay，每Graph32次完整core（含显式split/combine）；固定1秒热身、9交替配对、1/16输入，两边同模式，metadata规划/布局转换/H2D/编译/CPU FP64参考在Event外。16输入称轮转，不承诺cold cache。输出分配由Graph捕获固定，独立保存32份输出并在前后核对；输入GPU hash、offset guard、非默认stream另验。

首次SDK基线校准仅3公开anchors×2缓存策略×3新进程，不用于晋升。校准前冻结协议及脚本/合同/参考hash。每进程启动前要求连续3次GPU利用率0；忙则跳过未采样进程，不重跑已有Event。全部段保留；不稳定即失败，不挑选重采。AA单对<2%、组GM偏差<1.5%作为预注册保守要求；这是新MLA待实测门槛，不继承GEMM已通过结论。后续筛选最低GM1.05/区间下界>1.02/逐项>=.97，须先通过此MLA校准与候选AA。

SM89仅运行审查固定SDK/受控整数配置，尚无自由源码隔离，NCU不可用如实记录。新纯数值gate不承担身份鉴定；控制侧必须先验证protocol/source/SDK binary/environment/hash/完整QA绑定，不能凭mock KEEP激活发布。验证：`python3 -m unittest discover -s tests -p test_mla_measurement.py -v`；`python3 -m experiments.full_delivery.run_mla_paired --target 3070 --phase calibration --output <new evidence>`；server graph先做无Event探针。本记录先于MLA首段Event与新模型请求。

实施暴露两项绑定/调度故障：未显式TORCH_CUDA_ARCH_LIST导致SDK cache重建，首批324段仅诊断（数值PASS但身份拒绝）；显式架构+预编译后的SM86批又与本任务安装验证重叠，在查看数值门槛前整体废弃。v2唯一校正批增加同卡flock/阶段与Event时间戳，数值门槛/数据分布/段数不变。SM89首批无同卡重叠，可独立审计。全部旧批及汇总失败保留，不重采既有GEMM数值拒绝候选。

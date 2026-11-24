# HAGS实际启用后的独立环境资格

用户已确认3070重启。只读DxDiag实际HAGS Supported/Enabled均True；GPU UUID与驱动591.74不变，Windows bootUTC从2026-11-16T14:08:06.7925500Z变为2026-11-23T06:52:14.5000000Z。Docker服务及固定ce00f6dc镜像均正常，但旧/tmp dev6安装前缀已清空。

先复用归档dev6 wheel SHA d5f4bbf...和许可完整SM86 AOT zip SHA939bf2d6...，离线no-index/no-deps恢复到/home/dministrator/cuda-lab下的新项目持久前缀；不重建相同版本、不重新执行两类sealed验收、不读取精确清单。全部installed controls/AOT1456文件逐SHA验证。原来的1秒sustained qualification host、support_files、ClockControl与assess复用，原2%/1.5%/.97及32call×9pair×8workload×两策略×三AA/条件三AB不变。先短有限可信Torch活动核验HAGS环境下锁频实际状态，然后冻结包含实际HAGS/启动时间/恢复前缀的新环境身份。旧2.040408%身份永久拒绝，不换output重采。

来源继续固定KernelBench423217d9... CUDA Event/Graph、KernelAgent e0647170...、KDA acdd793...外置证据与状态；实际代码复用experiments/full_delivery/cublas_sustained_qualification_host.py及audit_cublas_sustained_qualification.py；无新CUDA算法或自定义ABI。

同时复核首个生产发布门槛：PLAN.md明确以SM86 GEMM的运行闭环/隔离/交付回退/运维为准，registry.index('qualified')与deployment.execute均无NCU运行依赖。旧文档将4090计数器权限和首个SM86生产准入并列过严，应区分为产品运行/首版准入与完整跨卡C研究。4090可运行可信SDK正确性/Event与有范围的普通调参；当前无Docker/user-net隔离，不运行自由模型CUDA源码。P6-03目标C保持未完成，不修改冻结研究协议、不将3070 profile当4090指标、不自动放弃全部完整计划；用户是否更换服务器尚未最终确定。

实施记录：collector原SHA5cf5bdf07eb79ab60e1ce037fbd7b02e0460a4e85466bdb05ad38234c74e2eab，只新增显式--installed-prefix并保留旧默认；SHA分段/重构和CPU审计原逻辑不变，解决旧/tmp已失效的导入路径。新增启动控制器在采样前核验DxDiag原件SHA/实际HAGS、同一boot、GPU identity和固定Docker镜像；原sustained host字节SHA7a10943be10156a18fcd49b85da941db36910ff52452cbb544d3947aa6232676不变。protocol ec6bbbc176a4c67df83bdc362feb3f95ce1cad1e73001db4a88f1692e947959d仅启动一次；远端/tmp/nka-cublas-final-hags-qb6rwyj0，pid3026。

自动审批拒绝了探针附带的持久bundle chmod0755，原因是持久可读范围扩大；操作未执行。已移除权限变更，纯8秒可信GPU探针通过，未改变目录权限。恢复的AOT根仍0700；若以后需要operator UID65534挂载，须另行按既有只读最小暴露设计处理，不把文件SHA恢复误称为重启后operator可运行验收。

验证命令：python3 -m py_compile experiments/full_delivery/{run_cublas_hags_qualification,probe_hags_clocks}.py tools/collect_segmented_public_run.py；/tmp/nka-p1-03-model-venv/bin/python -m experiments.full_delivery.probe_hags_clocks；同解释器 -m experiments.full_delivery.run_cublas_hags_qualification --output evidence/20261123-cublas-hags-qualification。后两者已消耗一次，不重复启动；后续只读收集并audit_cublas_sustained_qualification。

启动适配原始失败：后台Popen脱离SSH后原生nvidia-smi在锁频前UtilAcceptVsock accept4 failed110；21文件SHA归档978a861e84951b1be8e2afcccabb5e48a42a80cb450e69c3d7a8312429cc76bb，QA=0/Event=0。首次锁频及finally两reset均未执行成功，原始CLOCK_ENVIRONMENT_REJECTED保留；此前纯探针reset成功，后续前台live-parent续接两lock成功，未隐藏或假称首次恢复成功。新增resume_hags_unsampled.py只允许该归档证明的未采样状态，保留SSH/WSL父会话与WSL interop环境；相同protocol和host字节，非新身份重采。后续持久进度命令observe_cublas_hags只读。

CPU审计扩展audit_cublas_hags_qualification复核首次0QA/0Event归档、相同protocol/host、HAGS/boot/GPU/恢复/探针SHA，再直接调用原audit_cublas_sustained_qualification；不重新解释或放宽原数值规则。首次0采样失败与实际续接分别归档，独立于原HAGS-off三个失败批次。

实际续接结果：320QA，一AA/288段后因连续空闲条件未满足停止，原始状态GPU_BUSY_UNSAMPLED_REMAINDER；AA0已有4组单对超2%，最坏dev06/ring16=5.1423710000404466%，不能完成资格，更不续跑/重采以消除该失败。原sustained审计器强制3AA且decision非null，不能审计“中途GPU忙”状态；新增独立partial审计版本逐字继承原归档/帧/NPY/参考/Graph/pilot/clock检查，仅接受1AA/0AB/null正式decision/GPU_BUSY原状态并强制不合格，不修改原assess或冻结协议。完整三AA未完成明确标记false，已采超限只报告为observed违例；不是完整三session稳定性结论。

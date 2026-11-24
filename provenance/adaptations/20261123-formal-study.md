# P6 正式协议实施前记录

固定五上游pins保持：KernelAgent e0647170... provider/metric schema/collector；KernelBench 423217d9... CUDA loader、QA和Event；KDA acdd793... 候选与外置证据/晋升状态；ForgeKernel b2c8ae8... 当前参照/评测保护与各方法独立历史；FlashInfer-Bench 15618a67... CUDA builder/model职责。实际GEMM继续固定dev5有界编译/双binary工具，MLA继续FlashInfer3f8317c...的原fa2 CUDA与显式split/combine。不能把流程参考写成整套上游都已运行。

小规模实际缺口已记录：提示未给显式优化目标/父延迟，历史只含编译/正确性/延迟，导致重复、改名、no-op。新增study_feedback仅补安全allowlist的最终verdict/原因枚举/重复范围/方法自身patch或配置/父延迟；保留旧prepare_channel与provider不改。B不调用profile loader、不收硬件/原ptxas日志；C另给真实源码/硬件/binary绑定数据和成本。4项CPU边界回归与小规模原36条审计分开。原实验不倒填为新反馈。

正式计划6公开search targets×A/B/C×3独立搜索×最多12提案（648上限，B/C最多432新API），另受全局700新调用约束，已用32。每分支新会话且仅自身历史。前三次连续重复/keep_parent或无效提案、首次全局筛选通过、12上限停止；A遍历有限合法参数后停止，不能为了凑12重复GPU采样。失败/重复均消耗提案预算。先冻结controller/协议/初始cache manifest/提示再调用模型。

GEMM search targets建议dev01/02/06（512³/1024³/128×2048×512），这些实际进入可修改的WMMA kernel；原dev03与dev08落入cuBLAS，保留为全局保护，不能把不可变kernel的搜索当实质源码对照。此项只选择公开开发搜索目标，不改原工程screening anchors01/03/08、八workload、数值或1.05/1.02/.97/三session门槛。MLA targets仍02/04/08。全部独立shape/seed未读。

每C搜索先单独采集初始父版本profile并计实际wall，profile不替代Event。新GEMM候选若追加诊断，须在已实测的UID65534容器内运行，绝不拿到宿主root；SDK受控参数可沿既有可信路径。profile失败如实unavailable，不能伪造/填0。目标4090无权限未解决，C重优化如未运行须明确，不冒称当前有跨卡硬件反馈结果。

全局cache必须按源码/参数/协议/软件/硬件/二进制严格绑定，已拒绝身份不可重采，其他方法best/提案不传递。实际cache构建成本与命中边际成本分开报告；独立搜索允许相同精确候选复用既有实测，但不把复用当新独立性能样本。最终冻结后才执行独立4项MLA与最后配对；默认初始实现若未变，不重采已拒绝旧AA挑通过。

预采集GEMM9份成功；MLA9份虽然3launch/数值通过，但Nsight子进程的架构环境转发少一项，实际JIT重建为b700597...；实测Ninja仍是sm86，不能将漂移归因于PTX，与Event已冻结cdc4不同，严格拒绝且全部原始报告/新DSO/编译flags保留。补传该公开架构环境变量，并采用原JitSpec AOT布局冻结cdc4/31be已验证字节，避免后续任何动态重编覆盖。只补9份受影响MLA，原GEMM9份不重采，所有旧AA拒绝不重采。加载/身份witness适配位于Event范围之外，Graph/Event/CUDA字节不改；只继承先前保守拒绝，不把加载改动写成新收益。

正式调用后只新增结果接纳/审计辅助工具，不改已freeze控制文件或协议。新source每个精确身份用O_EXCL reservation防止并发重复执行；reuse现有installed dev5 evaluator/有界容器fresh build/QA/AA判定。仅明确违反冻结合同的源码可静态拒绝；新增指针局部量或外提地址计算可能影响编译，不能仅因数学等价就免实测并拒绝。若只是已有变体的变量名改动须给对比证据，不继承另一个source的数值结果。构建失败按失败证据记账，不伪造QA或延迟。新增辅助验证命令：python -m experiments.full_delivery.formal_results；独立原始审计仍复用audit_reviewed_source。

MLA独立正确性准备：复用GEMM RC1的私有manifest commitment/一次性reservation/私有SHA归档模式，实际执行复用dev6 OperatorRunner.run及pipe_worker原字节、PyTorch CPU FP64 references.mla_core，既有dev6 wheel与licensed AOT cfae bundle不改。新增仅可信host把封存4workload×2seed转换为默认page16/split1请求及外部oracle；候选仅见本次输入/request，不挂载manifest、参考、私有检查或凭据。精确shape/seed和NPY只进.private与远端0700控制目录，公开仅计数/hash/误差总值。先冻结公开协议和evaluator hashes，不读私有值；执行与三session性能验收分开，失败不自动重试或倒填开发。工具路径/安装保留既有环境。待验证：prepare_sealed_mla（纯冻结、无私有读取）；run_sealed_mla（私有一次性实测）；audit_sealed_mla（私有CPU复核、只输出汇总）。

跨卡补齐：源MLA所有27组搜索均结束保留page16/split1；目标4090直接复用、page16/split2及page64/split1普通调参已实测。补唯一尚未测page1/split1，复用冻结run_mla_paired/mla_paired原字节与同八workload两策略三session规则；预先只读确认fa2 cfba14...、cascade b8050d...不变。普通调参空间预定page1/16/64 split1与page16 split2，不重采已拒绝身份；目标C因为ERR_NVGPUCTRPERM仍未运行，不能用源卡profile代替目标卡。源best与通用初始相同，直接迁移与通用起点重搜身份重合，按cache标注，不能捏造独立重复。

生产默认cuBLAS最终复测准备：实际发布默认artifact66fcc...和回退93fb...的入口是gemm_cublas，历史不稳定身份是WMMA模板/配置；新复测身份明确包含backend=cublas与binary hash，禁止借此改写旧WMMA拒绝。复用dev5/6的SourceRunner低权限边界、pair_worker.load_named/check和source_timing.paired_witness；另存qualification支持副本，只把两角色固定为cuBLAS及Event段按预先250ms目标、3次4-replay pilot中位数校准（1..2048replays），32call/Graph、9配对、3独立AA后条件3AB、全部8workload两输入策略、原2%/1.5% AA门槛保持。已有独立正确性16/16和回滚仍按旧freeze引用。该步骤是现有默认与回退的部署资格检查，AB原1.05/1.02/.97优化筛选原样复算，未达收益不晋升；额外报告部署回退保护项>=.97是否满足，不混同优化收益。先冻结公开protocol/source hashes，再执行一次；GPU同锁序列化，旧identity不重采。未完成前生产false。

首批GEMM生产通道准备：复用现有Registry.initialize/release/fallback/rollback和deployment.execute/DockerRunner，不替换Torch绑定、不改CUDA产物、不改legacy catalog或optimization bundle。实际缺口是旧catalog明确仅development且代码拒绝production metadata；新增显式qualified通道，资格证书/新catalog仅在独立16/16、恢复、三AA/三AB及原始CPU审计全部符合后生成。默认仍development；qualified材料缺失时在GPU前拒绝。新增包用独立0.1.0rc1版本，不覆盖dev6 wheel。改动前源码快照evidence/20261123-qualified-channel-source-delta/originals。验证：CPU缺证书拒绝/manifest绑定/旧通道兼容/回滚；资格成立后新prefix空cwd实际安装和qualified输出/故障恢复/回滚。该通道声明范围仅SM86固定image的cuBLAS GEMM，算子与全文项目生产状态不随之自动通过。

cuBLAS资格AA实测失败4.274%，不生成qualified证书、不执行AB、不重采。原flat whole-file collector在120s传输前超时；真实GPU运行已完成，远端原始目录保留。补充只读归档复用experiments/gemm_rel01/archive_parts.parts（NKA1/ZIP_STORED成员无损SHA分段）和run_feedback_evaluation已验证的blob去重/整体SHA重建；新增collector仅目标精确public /tmp/nka-cublas-final-*目录，128MiB单文件cap，绝不读取sealed目录或再执行GPU。失败收据保留；一次更长有界传输不算性能复测。验证：collect_segmented_public_run、CPU逐文件/分段/帧/NPY/外部参考receipt复算。

资格AA失败后rc1只作为未发布预览：不生成真实证书，qualified命令必须GPU前拒绝，development通道可继续验证。新增安装delta复用pip no-index/no-deps/新prefix空cwd/原应用symlink回滚模式，实际运行相同两份cuBLAS artifact；验证5输出、crash/timeout各一次恢复和rc1→归档dev6回滚，零Event/零模型。19项相关CPU测试在新安装包命名空间执行。该验证不覆盖/重采失败资格，不标生产通过，不改dev6 wheel。

WMMA逐lane指针提案静态拒绝已对照NVIDIA官方CUDA编程指南核实：256-bit对齐、所有warp lanes相同mptr/ldm/layout。https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/05-appendices/cpp-language-extensions.html 。原static-review proof不修改，核实另存reference-verification.json。

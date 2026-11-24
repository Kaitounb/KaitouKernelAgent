# SM86 Attention / MLA 发布补齐

实施前：沿用 provenance/upstream-lock.json 的五个固定上游，及既有已验收 PyTorch v2.7.1 PersistentSoftmax / FlashInfer v0.2.8 fa2 适配和精确 AOT bundle cfae38ce5c0fac56545207fe93588c1193646886fc5952f113fab487ef0be6eb。核对的成熟模块为 FlashInfer 的 single_prefill_with_kv_cache / BatchMLAPagedAttentionWrapper 与本项目 flashinfer_backend.attention / SplitMLAPlan。数值适配、partition、CUDA .so 均保留，不新增 kernel 或 ABI。

具体缺口：既有 operators.cli 只有 development-only，未绑定真实证据证书、固定GPU/驱动/image及包内运行文件；postboot恢复的原bundle目录0700使低权限worker不可读。复用既有 OperatorRunner / bounded_io.capture / protocol / runtime_contract / AOT哈希目录与原子输出，不重写执行框架。只增加qualified门禁、环境绑定与真实容器边界witness，并在独立公开目录复制相同SHA字节，保留原恢复目录与权限。

验证：新0.1.1rc1版本只构建一次，固定image内不联网不JIT，在新prefix/空cwd做CPU门禁、实际Attention/MLA正确性、逐算子crash/timeout/fresh恢复、实际包rc1→已归档dev6→rc1切换与原始帧/NPY独立FP64复核。未取得全部真实证据前不打包资格证书、qualified在GPU前拒绝。已消耗sealed与旧A/A失败不重跑，新Event必须有实质新环境/目标、先冻结协议且GPU独占；本批不声称性能收益。

实际新增边界witness首次使用嵌套对象/整数列表，与既有扁平NKA1格式冲突，GPU计算完成但两worker均在encode_frame拒绝。未放宽protocol.py，改为boundary_*标量、字符串capability/版本，原失败及source快照留档；新冒烟2输出及正式安装13输出/17尝试通过。37CPU和五GPU前拒绝通过；本机NumPy FP64重算15成功输出PASS，原四数值/合同文件和1456 AOT文件不变。运行时本地源码版本0.1.1rc1，唯一wheel93b228...，旧版本不重建。

计时复用experiments/full_delivery/mla_paired.py的两CUDAGraph、32节点、缓存1/16、9配对轮换与guards/前后QA；当前新目标是HAGS-On固定Attention/MLA部署，增加Attention公开锚点，冻结pilot公式将目标段设1秒，仍用原operators.measurement.assess的2%/1.5%门槛。固定成熟SDK在可信控制层执行并使用CPU参考，不宣称其为恶意源码隔离；实际部署Docker边界由前项实测分别证明。protocol e435475828b5909d78901b97c22a499c23e901cfd6cbf89412ee755708a47d2a，0新API，不重跑旧拒绝或sealed。

原三AA完成540段/541.092 Event秒，最大单对5.598675%、组偏差2.162404%，540段及840文件独立审计通过但资格拒绝、0AB。Attention-decode/count1的mean在session0/2约0.983/0.980、session1约1.022，随capture order翻转；对应loaded时钟均1935MHz，不能把偏差简单归因于锁频或用户后台。发现原source_timing.paired_witness共享一组GPU input ring，旧MLA跨page配置适配为了布局不同而分配两组；本轮两侧同一布局却保留了分配差异，存在缓存/地址混杂。采用原成熟shared-ring方法，保留两Graph/同缓存策略/2%及1.5%门槛与1秒pilot，增加两侧真实device-pointer相等见证、测前后hash/guards。只预注册一次方法更正批，旧拒绝不改，不能把“地址是唯一原因”当已证明。数值/运行包未改，不重跑sealed或模型。

静态发布审查补出具体入口缺口：/sdk在Python startup的PYTHONPATH中，仅核验manifest列出的文件不能拒绝额外sitecustomize.py或伪造__pycache__/*.pyc。增加主机GPU前完整文件集合/无symlink与special entry检查，沿用read_regular有界manifest，不改worker/数值/计时。两项CPU安全回归分别覆盖额外startup/bytecode及symlink。已归档rc1不改；工作树版本进入0.1.1rc2待实测，正式资格只在全部真实证据通过后发行0.1.1。

最终来源固定与保留：FlashInfer v0.2.8 commit `3f8317c67ac129facb8525b2fb581be42f634459`，`prefill.py::single_prefill_with_kv_cache` / `mla.py::BatchMLAPagedAttentionWrapper` / `include/flashinfer/attention/mla.cuh`；CUDA kernel、调度、LSE转自然对数、split空行处理、4个数值/合同文件及原1456文件AOT均不改。PyTorch v2.7.1 source `e2d141dbde55c2a4370fac5165b0561b6af4798b` 的PersistentSoftmax旧开发实现保留，本次qualified不包含Standalone Softmax。KernelBench/KDA等五上游pin未变，非全框架重写。原source_timing.paired_witness共享ring的既有成熟实现用于纠正计时混杂；固定SDK性能控制器用可信FP64参考，与实际低权限Docker运行边界证据分别记录。

更正protocol95fefcec...三AA三AB1080段/1080.726 Event秒全部完成，最大AA单对1.935923%、组0.447360%，AB最小0.999478866×；1668文件和1632数组独立NumPy FP64审计PASS。原540段5.598675%拒绝不改。Attention新服务端合成清单commit7f0159d6...、protocol8a3ccbef...，冻结rc1后一次性8/8，239私有文件/NumPy FP64复算PASS，旧私有数据完全不读不传不重跑。

rc2单独不可变wheel6bb385a9...，实际40CPU/2输出/四目录拒绝PASS，4773文件审计。OperatorRunner.run与cli.execute AST及worker/数值文件与rc1一致，只增加入口准入/展示；故障恢复证据由原rc1和新GA实测分别证明。正式0.1.1的12运行文件与rc2逐字相同，仅版本/真实证据证书新增；证书261b6fcb...、wheel f2ef0ec1...，实际40CPU、9输出11尝试（7qualified+2预览回退）、五真实资格负例、GA→rc2→GA包切换均通过；5160文件/帧/NPY/NumPy FP64最终审计PASS。持久current恢复GA，GPU/driver/image与全文件校验生效，startup/pyc无法绕过SDK完整集合。

本轮0新DeepSeek/0新CUDA编译/0系统设置变化/0旧sealed复跑；未对外注册表发布。源自实际证据的生产范围为SM86_REVIEWED_FIXED_ATTENTION_MLA，普通时钟/HAGS On下Graph批平均，不冒称优化收益或单请求尾延迟。全部项目GPU/API已结束，桌面SMI15–18%/1058MiB留实测，不称整卡空闲。

已完成验证命令：audit_sm86_operator_release、audit_operator_deployment（原/更正两份只读）、audit_attention_independent、audit_sm86_operator_rc2、audit_sm86_operator_qualified；真实GPU入口已消耗，不重启run_operator_deployment(_shared)/run_attention_independent/install同版本。原始证据见TASKS及releases/sm86-operators-v0.1.1/manifest.json。

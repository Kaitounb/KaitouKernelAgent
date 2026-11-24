# SM86 GEMM 热条件诊断（非资格重采）

2026-11-24：ordinary 批次三 AA / 864 段最坏 7.699345%，独立审计通过但资格拒绝。负载遥测最高 88°C、SM 1665–1950MHz；结束后查询目标温度83°C、功耗默认240W，支持100–240W。累计热降频计数不等于本批增量，不能据此证明因果。

复用 provenance/upstream-lock.json 的原 KernelBench CUDA loader 与已安装 source_worker.load、原cuBLAS二进制66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3；不改 CUDA/数学/输出合同，不用旧sealed。参考 NVIDIA 官方 https://docs.nvidia.com/deploy/nvidia-smi/ 的 -pl 支持边界；100W选硬件报告最低允许值，避免功耗搜索。先只做一次设置/读回/恢复能力检查；支持才做一次有界温度诊断，不输出部署通过结论。原240W必须finally恢复，保留命令返回和全量遥测。

诊断若支持：固定公开1024³、32次cuBLAS Graph连续90秒；100W上限，不锁频；开始/结束查询POWER,TEMPERATURE,PERFORMANCE,CLOCK，记录热降频累计计数差。诊断只用于判断热修复是否可行，不读取AA对结果、不改变3%/1.5%规则、不补跑失败身份。

验证命令：python3 -m experiments.full_delivery.gemm_power_probe（一次性控制器，证据 evidence/20261124-sm86-gemm-thermal）。现有独立验收与全部原始失败保留。

首次容器执行未覆盖固定image ENTRYPOINT，默认NKA1 worker在空stdin上JSON解析退出1；未导入target、0负载，100W已finally恢复240W。原失败留档，修正为显式--entrypoint python后执行原90秒公开热诊断，无AA身份重采。

诊断完成：90.129秒/31328 Graph replay×32 calls，最高54°C/99.98W，软件与硬件热降频计数增量均0，数值max_abs2.636897e-5，恢复240W。此证据仅支持一次预先冻结100W动态频率部署条件评估，不证明AA通过。沿用KernelBench 423217d9fda91e0c2d67e4a43bf62f96f6d104f1 / src/kernelbench/utils.py::load_inline绑定及现有source_worker.load；保留两cuBLAS binary/数学/共享ring/计时方法/三AA后才AB/所有阈值，仅新增外层同GPU锁内100W设置、三次读回、全程限额与<83°C/热计数不增门禁、finally240W恢复。evaluator gemm_power_host.py；一次批次通过或拒绝均不重试、不调整上限搜索。

0.1.2rc1候选控制入口追加100W/起始温度<83C硬门禁与证书operating_profile完整匹配；真实证书仍待性能与安装证据，不因CPU模拟签发。保持registry/deployment/runner/pipe_worker等原证书绑定文件逐字不改。验证 unittest tests.test_gemm_target_binding，增加240W/临界83C/NaN/缺失或错误限额拒绝。

预览安装复用已通过的install_sm86_operator_release离线pip wheel/新prefix方案与sm89_installed_validate恢复/实际包回滚检查，新增GEMM通道入口与两既有算子冒烟；先构建0.1.2rc1一次且不覆盖旧版本。49CPU本机通过；安装验证先等待同/tmp/nka-reviewed-gpu-execution.lock，不能与正式计时GPU重叠。公开collector只添加精确gemm-0.1.2(rc1)-持久目录白名单，不扩大私有目录访问。命令python3 -m experiments.full_delivery.install_gemm_release --phase preview；原始失败同样保留。

后续GA安装验证（预览已上传快照不变）预设同卡锁内100W准入、240W真实错误条件拒绝、运行policy文件篡改拒绝、finally恢复240W；仅在真实资格签发后执行。安装控制器等待GPU锁允许5400秒，避免三AA/AB完成前排队超时；已启动预览原1500秒记录不改。签发器仅当100W三AA+三AB1728段/独立审计/新预览11输出13尝试全部通过才可执行，核旧独立freeze、两binary/合同/worker/protocol/output逐字继承，不再读或执行sealed。

完整100W三AA/864段仍拒绝：15.0113948%，组1.3875418%，最高55°C、热计数无增、240W恢复。预览49CPU/11GPU输出13尝试/592文件独立审计通过，不签生产证书。追加一次54个段的有界small-M native Nsight时间线，仅定位CPU enqueue/GPU Graph停顿，明确不同于Docker正式样本，profiler开销不能转成新资格。直接沿用run_cublas_timeline_fixed的专用HOME/时间戳修复/原profile/export命令和原cublas_timeline_target的已核二进制/共享ring/CPUenqueue-wait分解；只缩小到公开dev03/count16，加固定5秒1024³预负载、1秒段和wall时间便于遥测对应。无参数搜索、不读取私有清单、不改失败身份或阈值；结束恢复240W。入口run_gemm_power_timeline.py，源码及Nsight原始证据保留。

时间线初次新增预负载遗漏capture外cuBLAS handle预热，CUBLAS_STATUS_NOT_INITIALIZED在Graph捕获首调用失败、0个诊断计时段，原trace/stderr保留且恢复240W。更正仅添加8次capture外调用和stream同步，再执行原定一次诊断；不重采正式AA。

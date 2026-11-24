# P3-02 有界运行通道与部署恢复（实施前）

2026-11-22，Asia/Shanghai。已读当前README/HANDOFF/TASKS/PLAN；P1-05计时未验收，不再次重采，不改变合同／kernel／搜索门槛。当前P3-02原runtime有宿主可写outputs挂载、stdout/stderr无限文件、候选JSON直接读取三个具体缺口。

| 固定上游／符号 | 保留／复用 | 必要适配 |
|---|---|---|
| FlashInfer-Bench `15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee`, `isolated_runner.py::SubprocessWorker.run_solution` | 单worker、输入交接、成功／错误、外置timeout/finally清理生命周期 | 原同UID/Pipe传Python对象与ref_outputs，不能直接作为本项目不受信边界；不传参考／门槛，Docker固定image、两只读bind，无宿主可写mount。stdout固定长度header+JSON+NPY，控制侧先限长再解析，无pickle |
| 同版本 `data/solution.py::Solution` | 来源／依赖／目标硬件／source identity可追溯，immutable artifact身份 | 后续安装／版本指针以独立artifact SHA和合同／image约束绑定，不能把worker自报成功当发布授权；不改上游模型或宣称完整FlashInfer系统采用 |
| KernelBench `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` 已接入CUDA路径 | 原CUDA、Torch extension、独立CPU FP64参考；原`supervisor.load_output` NPY防symlink/shape/dtype/header/pickle检查 | 该函数提取到不导入Torch的输出模块，保留检查；反序列化在trusted侧。复用已归档的新旧可信binary，不再编译、生成源码或计时 |
| ForgeKernel `b2c8ae8510e02c28576808910a6b8827721ab7dc`, `harness_guard.py`; KDA `acdd79343c5e9d0c7cb96289f26fd3f1051ce785`, `docs/agent-flow.md` | evaluator／合同／候选与发布状态由可信控制侧持有；证据和父版本链 | 候选只看到所需输入／固定binary／worker，不挂项目、凭据、参考、发布目录。回退／回滚将独立演练，开发演练不当正式晋升 |

先实现受限字节通道：Docker `--log-driver=none`，stdout/stderr selectors逐块读取，写入前限总字节；超限/timeout只清理本次UUID容器，保留有限前缀及状态。容器outputs为8MiB tmpfs，tmp128MiB、只读root，文件大小上限16MiB；PID32、CPU2、CPU内存2GiB（内存故障fixture单独256MiB，不导入Torch）。无宿主RW bind。NPY大小按可信shape决定，metadata最多8192字节、flat JSON，无候选路径回传读取。

预定验证：

```text
python3 -m unittest discover -s tests -p test_bounded_io.py
python3 -m experiments.p3_02.run_bounded_3070 --evidence <new-dir>
python3 -m experiments.p3_02.audit_bounded --evidence <dir> --output <new-audit.json>
```

GPU回归只做正确性与恢复，不出性能结论；固定fixture覆盖正常cuBLAS/模板、常数、污染、异常、错误frame/metadata/NPY、stdout/stderr超限、容器tmpfs满、PID上限、CPU内存OOM、有限CUDA后host超时，每次故障新容器cuBLAS恢复。fixture有界，禁止无限GPU kernel和无限fork；不触碰用户其他容器。完整生成源码／builder产物／日志边界仍需单独验收。安装／版本回退和回滚实施前追加具体命令与范围。

## 安装与回退／回滚实现前追加

标准setuptools／pip wheel打包Python运行适配器、原公开合同及两个已实测trusted binary（93fb2f…与66fcc8…），保留NVIDIA原版权／许可。上游构建仍是KernelBench+Torch，非新FFI；FlashInfer Solution来源／依赖／硬件身份原则用于安装manifest，不执行其完整模型或伪称采用原部署API（上游没有本项目Docker版本切换接口）。

仅开发部署演练：两个版本均只启用cuBLAS，原4-warps搜索freeze不变、不晋升候选。manifest绑定binary、合同、image、Torch/CUDA/SM86，完整性能／独立验收仍未通过，安装状态明确development-only。可信侧持有版本目录和原子current/previous指针，worker不挂载它们。独立保留旧cuBLAS fallback binary，活动版本缺失／损坏／元数据不匹配或运行异常／超时后最多新起一次fallback；fallback失败则退出，不重置GPU、不循环重试。显式rollback只切换验证过的版本，含锁与原子replace，不回滚用户其他文件。

预定命令／验证：

```text
python3 -m unittest discover -s tests -p test_registry.py
python3 -m experiments.p3_02.install_regression --evidence <new-dir>
# 目标已有固定Docker image及NumPy，包构建／安装禁网络：
python -m pip wheel --no-deps --no-build-isolation --wheel-dir <dist> .
python -m pip install --no-index --no-deps --target <new-prefix> <wheel>
python -m kernel_agent.cli init --root <new-runtime-root>
python -m kernel_agent.cli run --root <root> --a <a.npy> --b <b.npy> --output <result.npy>
python -m kernel_agent.cli rollback --root <root>
```

从wheel在新prefix／空cwd执行，PYTHONPATH只指安装目录，禁止依赖旧项目或/tmp代码路径。独立host CPU FP64检查正常、缺失／binary损坏／manifest不兼容／host异常／有限CUDA后host超时的实际回退，以及显式A→B→rollback的binary身份及输出；保留失败和原始结果。此为声明环境中的实际安装／恢复演练，不等于正式独立验收、生产发布或任意源码执行。

首批新通道运行已完成5项但metadata发送失败：PyTorch的torch.__version__为str子类TorchVersion，严格flat metadata发送器只接收内建str；未放宽接收边界，只把可信sender版本转成str。失败源码／原始stderr／五容器清理留在evidence/20261122-p3-02-bounded-runtime/，新目录重试。

第二批新通道通过前17项（正常／边界／常数／污染／异常／frame／metadata及恢复），stdout_flood时Docker rm超时。原capture在关闭未读attach管道前调用rm，输出背压阻塞清理；失败日志留在evidence/20261122-p3-02-bounded-runtime-version-fix/。修正为先关闭两管道、终止本次docker客户端，再执行owned容器rm；控制命令超时记unknown并fail closed，只有明确No such object才确认容器已删，不再把任意inspect错误算成功。新目录重跑同一固定组，原限额不变。

第三批所有29个容器的限额／行为已执行，但Docker29返回小写`error: no such object`被大小写敏感确认器误判为未删；同时默认sig-proxy将终止client的SIGTERM传给worker，虽已退出但无法在rm前抓到活着的child，超时证据不足。失败完整保留cleanup-fix目录。确认器改为不区分大小写、仍要求exit1且明确No such object/container，daemon断连／timeout不算成功；增加CPU负例。Docker run显式sig-proxy=false，使本地client结束不替代owned容器的显式清理，按top记录host pids后再rm并核对消失。原限额／29项不变，新目录验证。

最终本阶段实际结果：confirmation-fix目录固定29项PASS，12故障拒绝／12恢复；wheel离线新prefix安装，15次独立正确性、五回退场景和实际rollback通过，3个CLI负例、目标4项CPU mocked-worker策略检查分列。wheel SHA dd8839f2bd723bbc20879890d997e01407ccf2fe6c9153a558d77a8c857bbbdb。默认cuBLAS、development-only，原best／合同／CUDA不改；尚未开放生成源码，builder产物／日志上界与完整源码联调／独立验收／生产发布待验。

收尾静态审查发现capture先Popen、后创建日志文件，若日志创建失败可能留下未受管worker。将日志／selector准备移到Popen前，部分创建失败也关闭已有FD；注册步骤纳入finally保护。新增真实目录占用故障＋Popen mock证明失败时完全不启动worker；不以它代替GPU故障测试。开发包版本升0.1.0.dev1，新证据目录重新离线wheel安装及同一15次正确性／回退／回滚；原dev0包及全部实测留档不覆盖，frame／限额／CUDA／合同不变。

同一收尾审查补齐写日志／close失败仍执行cleanup，以及cleanup命令异常保留removed=false而不是丢失状态后启动fallback。新增两项CPU故障注入，合计16项I/O／协议检查。最终开发包升0.1.0.dev2；先前dev0/dev1实测都保留，dev2将用相同安装／15正确性／回退回滚与4项策略组验证。仅改变host I/O错误路径，已有frame／GPU限额／CUDA／合同不变。

最终dev2实测／审计PASS，wheel SHA 877f7d6f31fcb2f6830397a3f6846831b009411559c7ffd7d06b738e3a2f46ea。目标新prefix导入路径 `/tmp/nka-p3-02-install-jpfp6x5q/installed/kernel_agent/__init__.py`，离线构建／安装、15次独立正确性、五回退／显式回滚、3个CLI负例、4个CPU mocked策略测试全部保留。当前本机I/O／协议测试16项、registry10项；29项资源故障实测仍引用原冻结runtime，本轮dev2无CUDA／frame／限额变更。完整P3-02仍未勾选，下一步builder有界产物／日志及源码入口联调。

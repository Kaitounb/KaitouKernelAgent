# P3-01 真实源码构建与新证据适配（实施前）

2026-11-22。已审查真实DeepSeek补丁：仅把WMMA K循环改成两次16-wide load/mma及单片尾部，无新barrier／host代码，license/includes／host launcher／cuBLAS／合同均不变。Git --recount只恢复hunk计数，原FAILED usage及原响应保留；总模型调用仍1，2,232tokens。

| 来源／固定版本与符号 | 复用 | 必要改动与验证 |
|---|---|---|
| KernelBench `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`，已验收Torch loader；本项目gemm_regression.make_inputs/guarded/check_case | CUDA构建和原输入生成／guard／producer-consumer stream／fresh output检查 | 构建继续P3-02 DockerBuilder；make_inputs/guarded逐函数提取且保留原体。参考输出只由可信host在调用前生成，child只回输出NPY、输入／guard hash，不计算或接收参考 |
| 已验收gemm_candidate.paired（KernelBench适配） | 原32节点／12 replay／9对、Graph捕获顺序、1秒warmup、Event交替段原体 | 新源码需独立A/A；移除child中CPU答案计算和比较，返回实际GPU输出hash／输入hash。host用先前独立FP64验证过的输出hash ledger核验，ledger不发child；未知hash即拒绝计时证据，不把它当数值错误。仅验证受审查的pure-device变体，不宣称任意恶意native计时防伪 |
| KernelAgent `e0647170da36ef9b059ac0bd3d60103aa4ed378b` | 原provider，原response／usage／finish_reason | 本次不再调用API或profile；只使用已返回补丁 |
| KDA `acdd79343c5e9d0c7cb96289f26fd3f1051ce785`、ForgeKernel `b2c8ae8510e02c28576808910a6b8827721ab7dc` | source／parent／artifact／contract绑定，外置裁决，固定harness | 不修改9个受保护原文件或旧P1-05裁决；模型源码另存，构建成功不晋升，计时不稳则保留best／cuBLAS默认 |
| FlashInfer-Bench `15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee` 生命周期＋已验收P3-02 runner | 低权限、两只读bind、有界管道、tmpfs、外置超时／cleanup | 新source QA批次的标准ZIP_STORED/NPY固定文件集合按可信case推导；不extractall、不pickle、不暴露expected/ledger/key。新.so只在child导入 |

预注册：候选先fresh build；8开发workload×seeds171/172、原6边界×2、另K=16非零和K=48尾部×2，共32项合同检查；graph seeds701..716×8共128输出由host FP64校验，建立ledger，总160项。每项验证输出FP32/shape/finite、新storage、明确producer/consumer stream、输入bit hash及offset guard。可用时以同一隔离边界跑memcheck/synccheck小边界，失败／不可用保留。

GPU连续3次util=0才运行新identity A/A：3独立session×8开发项×重复／ring16×9对，共864原始段；每session测前／后32合同witness核验，未通过就不跑A/B。不扩大单对2%／group1.5%界限，不因结果再次重采同批；旧4-warps P1-05失败不变。A/A全通过后才实现／执行双binary A/B，沿用原1.05／1.02／0.97筛选门槛；当前阶段未具备完整A/B不得KEEP。所有正确性／失败／busy／缺失保留分母。

预定命令：

```text
python3 -m experiments.p3_01.run_source_3070 --proposal evidence/20261122-p3-01-source-recounted --evidence <new-dir>
python3 -m experiments.p3_01.audit --evidence <dir> --output <new-audit.json>
```

P3-01只在真实源码结构变化、构建／正确性和失败证据明确后按任务范围验收，不等同于性能优化成功或生产发布；4个封存任务不读取、不执行。

首次真实执行160项正确性全部通过；两项Sanitizer退出0，但默认stdout banner混入NKA1，严格解析拒绝，未启动A/A。原结果完整保留。宿主已安装工具help确认`--log-file`默认stdout；只加`--log-file /dev/stderr`，要求stderr明确`ERROR SUMMARY: 0 errors`。新supplement目录复用同一SHA binary，CPU重审原160项NPY／FP64，不重复GPU QA／编译／API；原A/A未开始，仍只允许预注册一批3session。容器权限、计时与判定阈值不变。source_packets额外拒绝非对象report，6项CPU封包拒绝测试通过；此项不改变正确输出协议。

真实补验：同一binary的memcheck／synccheck各8项通过、明确0errors；宿主重审160原输出。独占条件满足后只跑一批3session／864段，session2 dev03/ring16偏差3.2329586%超2%，拒绝且不进入A/B。完成测量后汇总files变量NameError，原stderr、progress与3份完整report保留；修正为显式build.source_identity，新增CPU失败分支回归，不重跑GPU。独立审计从raw段重建裁决。21项源码相关CPU测试通过（10patch／4mock provider／6packet／1汇总），mock不代替实卡。

原始大包文本回传未完成；只读文件归档恢复首次SSH connection closed失败留档，未重跑候选／API。新fetch_archive入口只把已完成原记录归档到专用Windows临时文件，再SFTP复制校验SHA，内容和原manifest一致；不清除远端失败／raw记录，不触碰用户VSCode连接。

最终归档完成：原连接完整返回267,662,579-byte ZIP／248原始文件，本机source/artifact/NPY/witness/lifecycle／864 Event段审计PASS。文件恢复两次connection closed、专用转发strict hostkey失败均保留，未放宽主机密钥校验；专用转发已关闭，原3070 helper已恢复原验证版本，未触碰用户VSCode连接。source-gpu-file-transfer*为失败只读恢复尝试，不是新增GPU运行。独立decision.json为REJECT/candidate_aa_unstable；acceptance.json只验收P3-01源码结构修改／新证据拒绝范围，性能与生产发布均false。

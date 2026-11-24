# P3-01：真实 CUDA 源码提案与拒绝闭环

2026-11-22，RTX 3070 / SM86，固定CUDA13.0／Torch2.14.0+cu130镜像。首个真实DeepSeek源码候选完成隔离构建、独立正确性和Sanitizer检查。新源码自身A/A不稳定，裁决拒绝；未运行A/B、未晋升、未发布。学习复盘继续延后。

## 模型和源码

复用固定KernelAgent provider，`deepseek-flash`实际调用一次，1,641 input +591 output=2,232 tokens；最大输出768、60秒、零重试，finish_reason=stop。API未返回金额，不推算实际账单。加上前两轮参数提案，累计三次调用8,790 tokens。密钥只在本机控制进程环境，不归档、不复制到远端。

原JSON合法，但diff hunk行数错误，严格Git校验失败；原[usage](../evidence/20261122-p3-01-deepseek-source/usage.json)仍为FAILED。使用Git原生`--recount`另存派生源码，只恢复hunk计数，不修改补丁源码行、不重复API调用。原响应SHA为`ad84460ae9362c971266911d1f000c0392ba9fcb237416fec62b56ef8b78506c`。

真实结构变换为WMMA K循环每次处理两片16-wide tile，再处理单片尾部；保持FP32累加次序。license、includes、host launcher、cuBLAS和合同均未变。可信控制侧审查通过后，才进入已验收的P3-02 builder／GPU worker。

| 绑定 | SHA256 |
|---|---|
| parent／当前搜索参照 | `f329ade329014e517bf5e3b3a5828f7bc1b1c86102ffc310a6519110e85426a6` |
| 新candidate | `7fd6a3a1ecf75c4e56bda79c8b223358a7cdab96e7d0e5e8840b59926ec36d12` |
| 新WMMA源码 | `2ec37a08a96b374eac5038af49e81c987ad0dd74f4822b264f5503f9a74773d2` |
| 三CUDA文件identity | `41979ab009208d172d55d058db9841bdcb9284af5bf0c6aa0fabb68f1d4c89e7` |
| fresh build产物 | `9056a49cdb2e5b6ed8554dd3e89d53fb768a4823dcfae4f75c455c8c9dc3938b` |

## 正确性与隔离

固定无GPU builder从空cache新编译，未沿用旧源码成绩。独立非root GPU容器执行160项：8开发项×2seeds、原6边界×2、额外非零K=16与K=48×2，共32项合同；8开发项×16 graph seeds，共128项图输入的独立输出验证。包含FP32输出、连续布局、fresh/non-alias输入、显式producer/candidate/consumer stream、输入bit hash和错位指针guard。预量化FP16输入的CPU FP64参考由可信宿主在调用候选前生成，参考和验证输出ledger不进入候选容器。

补验复用同一binary，在可信宿主CPU侧重新核验全部160项，不重新运行GPU正确性批次。新源码memcheck／synccheck各8项边界、输出再经独立FP64比较，两者明确`ERROR SUMMARY: 0 errors`。

首次Sanitizer默认stdout banner混入NKA1，退出0但严格协议拒绝，原记录保留且未启动A/A。补验仅加官方工具`--log-file /dev/stderr`，容器权限未放宽。runtime仍无网络、非root65534、只读根目录、两份只读bind、cap-drop、no-new-privileges、PID32、2CPU／2GiB、128MiB noexec tmpfs、有界管道及外置超时／清理。普通容器不等于GPU驱动隔离或无限kernel取消能力。

## 新源码计时裁决

GPU利用率连续三次0后，执行预注册的一批：3独立session×8开发workload×重复／ring16×9对，共864原始Event段。原32图节点、12 replay、1秒warmup及交替Event路径AST与既有计时实现一致；每session前后32项合同输出／输入witness与已独立FP64核验的ledger匹配。图中32次输出的hash也与各自验证输出匹配。仅用于此次审查后的pure-device变体，不宣称任意恶意native计时防伪。

session2、dev-03、ring16最大单对偏差`0.03232958605147429`（3.233%），超过2%；该组GM约1.002671。候选自身A/A **FAIL_STABILITY → REJECT**，不进入跨binary A/B，无收益结论，不追加挑选重采。原P1-05的3.324%失败也保持不变。

补验在全部测量结束后发生报告汇总`NameError(files)`，stderr与全部raw reports／progress保留；代码已显式从build记录取source_identity，新增CPU故障分支回归。裁决由独立审计从原始段重算，不能通过重新测量替换失败。

## 状态与后续

P3-01范围是实际结构修改、绑定、新构建／正确性、证据失效和拒绝闭环；不要求候选一定晋升。本轮无新profile、未读取／执行四项封存验收。当前cuBLAS默认和development-only dev2 wheel均未改，新candidate不进入artifact allowlist。

全部原始证据已取回本机；[独立审计](../evidence/20261122-p3-01-source-sanitizer-channel/verification.json)PASS，[外置裁决](../evidence/20261122-p3-01-source-sanitizer-channel/decision.json)REJECT，[P3-01范围验收](../evidence/20261122-p3-01-source-sanitizer-channel/acceptance.json)通过。下一步是把源码预算／构建／验证／拒绝过程接到可安装的离线优化入口，并明确停止状态和恢复；B/C输入开关可先做离线验证，不因本轮完成自动新增付费调用。


## 证据和复算

- [原始模型响应／usage](../evidence/20261122-p3-01-deepseek-source/)；[派生补丁／源码](../evidence/20261122-p3-01-source-recounted/)保留原FAILED状态及恢复来源。
- [fresh build／160项原输出](../evidence/20261122-p3-01-source-gpu/)含完整lossless transport.zip、SHA内容寻址blob与只读hardlink；原Sanitizer协议失败也在其中。
- [Sanitizer补验／三会话A/A](../evidence/20261122-p3-01-source-sanitizer-channel/)含原始stderr／NPY／Event段、汇总NameError、21项CPU测试和派生裁决。
- 两次只读文件恢复分别因connection closed失败；专用转发因strict主机密钥校验失败而拒绝，未放宽校验，临时转发已关闭。最终原SSH文本传输完成，完整大包校验通过；失败目录保留。不涉及新的测量或凭据归档。

```sh
python3 -m experiments.p3_01.audit --evidence evidence/20261122-p3-01-source-gpu --supplement evidence/20261122-p3-01-source-sanitizer-channel --output /tmp/new-source-audit.json
python3 -m unittest discover -s tests -p 'test_source*.py' -v
```

本机审计只重解析原始产物／NPY、hash-witness和Event裁决，不导入.so、不运行GPU；数值FP64验收明确来自独立可信3070宿主，并已在补验阶段CPU重新核对160项。默认dev2 wheel和部署allowlist仍是原版本。

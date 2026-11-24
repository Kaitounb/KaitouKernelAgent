# 可安装的离线 GEMM 优化入口

2026-11-22，GEMM-REL-01按开发交付范围验收。`kernel-agent optimize`已脱离仓库路径，可在新prefix／空cwd使用。默认cuBLAS和两个部署版本未变，候选不会自动进入运行时allowlist；正式生产发布仍待验收。

最终开发wheel：[0.1.0.dev4](../evidence/20261122-gemm-rel01-dev4-delta/remote/run/dist/nvidia_kernel_agent-0.1.0.dev4-py3-none-any.whl)，SHA256 `5a30f76189e56b1094591a1c802329f6e43c70a572674163bb447d190c0cdf49`。包内固定39份优化资源／模块，manifest SHA256 `64e5fde57570249a2f12234e7a00969fd837a52262957018245a84da8c495432`。dev2和中间dev3的wheel及证据均保留。

## 安装及一次离线裁决

目标环境沿用已实测3070/SM86、Linux/WSL2、Python3.12.3、NumPy2.5.3、固定Docker image／Torch2.14.0+cu130／CUDA13.0。运行时只需原已声明的依赖；源码构建／独立参考另需目标已有Torch、Docker/NVIDIA runtime及公共编译依赖。不自动下载镜像、SDK或驱动。

```sh
NKA_PY=/home/dministrator/cuda-lab/.venv/bin/python
NKA_PREFIX=/tmp/nka-offline-installed
"$NKA_PY" -m pip install --no-index --no-deps --target "$NKA_PREFIX" /path/to/nvidia_kernel_agent-0.1.0.dev4-py3-none-any.whl
# RESPONSE是已有DeepSeek原始JSON响应；RUN为新的私有控制目录。
PYTHONPATH="$NKA_PREFIX" "$NKA_PREFIX/bin/kernel-agent" optimize --run-dir /path/new-run --response /path/response.json --recount
PYTHONPATH="$NKA_PREFIX" "$NKA_PREFIX/bin/kernel-agent" optimize-status --run-dir /path/new-run
```

默认严格Git应用；`--recount`只在严格hunk计数失败后显式使用Git原生恢复，原拒绝写入`patch-validation.json`，不修写模型源码行。模型响应／原usage保留，导入响应的新API次数是0，历史tokens单列。当前P3-01候选identity已绑定真实A/A失败，默认返回`REJECT/known_candidate_aa_unstable`，0新GPU测量，不用旧成绩声称新收益。

## 审查与检查模式

未知候选先保存`NEEDS_SOURCE_REVIEW`，退出码3。控制侧审查CUDA语义后提供精确WMMA源码SHA；`--review-sha`是审查结果绑定，不接受模型自己宣称已审查。固定单CUDA文件、parent、license/includes、host launcher和其他合同均受guard保护。

```sh
# 仅构建、独立正确性和Sanitizer，不计时、不晋升；可以维护检查已知拒绝源码。
PYTHONPATH="$NKA_PREFIX" "$NKA_PREFIX/bin/kernel-agent" optimize --run-dir /path/new-check-run --response /path/response.json --recount --checks-only --review-sha SOURCE_SHA256 --deps /path/public-build-deps
# 若先停在源码审查：同目录resume，不重新调用模型。
PYTHONPATH="$NKA_PREFIX" "$NKA_PREFIX/bin/kernel-agent" optimize --run-dir /path/new-check-run --resume --checks-only --review-sha SOURCE_SHA256 --deps /path/public-build-deps
```

`--deps`只指向现有公共依赖，包含`pydantic`和`bin/ninja`；不能放模型凭据或私有数据。本次使用pydantic2.14.4、pydantic_core2.50.5、annotated_types0.8.0、typing_extensions4.16.0、typing_inspection0.4.5，Ninja复用目标原安装的binary。依赖文件hash逐一保留，低权限builder两份只读挂载、无GPU／无网络；宿主CPU参考、验证输出ledger、模型凭据及release store不进入候选容器。

普通模式只对新、已审查identity在正确性和Sanitizer通过、GPU连续三次空闲后执行预注册的一批A/A。失败即拒绝，忙时明确停止。A/A通过仍为`NO_PROMOTION_PENDING_AB`，在该dev4历史版本中双binary A/B裁决尚未接入，不能KEEP或激活产物。`--checks-only`明确`CORRECTNESS_ONLY`，不构成性能验收。

## 单次预算及停止状态

真实提案入口是`--deepseek --max-calls 1`，只读取控制进程`DEEPSEEK_API_KEY`，复用固定KernelAgent provider／OpenAI SDK2.8.1；可选依赖`model`仅声明SDK版本，不自动安装。使用固定deepseek-flash、768最大输出tokens、60秒、零重试。模型密钥仍留在本机控制侧；可把公开响应文件交给目标3070的离线入口。本轮未使用该入口新增付费调用。

同一run目录有flock互斥，原子写状态、fsync和trajectory。先持久化`calls_reserved=1`再尝试调用；中断后不因重入重置预算。真实请求未拿到可归档响应时，`new_external_model_calls=null/external_call_outcome=UNCONFIRMED`，只说明预算已经消耗，不能推断已计费。收到响应才记录1；mock和离线响应明确0。

- `MODEL_RESERVED`后中断：resume标为`STOPPED_UNCERTAIN_CALL`，不自动重试API。
- `EVALUATING`后中断：resume标为`STOPPED_NEEDS_CLEANUP`，不再启动GPU；先检查本次UUID容器／日志及清理状态。不是自动恢复无限GPU kernel。
- `NEEDS_SOURCE_REVIEW`：可以携带同一源码hash继续，改过源码或计时范围拒绝。
- 终态重复命令只读回已有结果，不重编译、不重测量。候选拒绝和检查完成退出0；待审查退出3；失败、未知调用或待清理退出1。

已有响应可在新的独立run中重放；这不是重置先前真实调用预算。已知A/A失败的普通优化仍直接拒绝，不通过新目录挑选采样。运行时`init/status/run/activate/rollback`沿用[原开发部署说明](SM86_RUNTIME.md)，优化命令不写release指针。

## 实际验收和证据

[dev3完整安装／GPU组合证据](../evidence/20261122-gemm-rel01-installed/verification.json)：新prefix、空cwd11条构建／安装／CLI命令；源码fresh build、160项可信宿主FP64／输入／guard／stream检查，两项Sanitizer各8项、0errors；一次默认cuBLAS独立正确性。历史hunk严格拒绝、显式recount、已知失败复用、终态幂等及错误预算拒绝通过。0新模型、0 Event段。

收尾14项CPU回归捕获empty choices的IndexError和调用预留误标，原失败测试留档。最终[dev4新安装／差异审计](../evidence/20261122-gemm-rel01-dev4-delta/verification.json)：空cwd13条命令，包含14项已安装CPU回归、坏响应拒绝、已知拒绝／幂等、无key预检和实际activate/rollback。与dev3的package逐文件比对，只有command、包版本和bundle manifest改变；其他构建／GPU／数值／Event／部署及公开资源逐字相同。dev4明确继承上述GPU组件证据，本轮没有在dev4重复运行GPU，也没有新增真实API调用。

原始frame／NPY／日志全保留；传输仅对ZIP_STORED成员分片，共享已有只读SHA blob，重建350个原文件后逐个核对整体hash。首次GPU证据仅传1,621,872 bytes，不是用历史GPU结论代替本次运行。3项传输CPU测试通过。原`.DS_Store`审计错误和SSH只读进度查询认证失败也保留。

上游provider／KernelBench原文件随包分发并附各自完整Apache-2.0／MIT LICENSE，具体来源见[实施前采用表](../provenance/adaptations/20261122-gemm-rel-01-installable.md)。NVIDIA CUDA来源和运行产物通知仍保留；KDA／ForgeKernel／FlashInfer-Bench按各自已声明的流程／生命周期借鉴范围，不宣称五套框架完整打包。

后续[dev5双binary组件](DUAL_BINARY_EVALUATION.md)现已按安装／固定夹具／诊断范围验收，普通模式已接三A/A→三A/B外置筛选；本篇dev4记录保持原样复现，完整新source协议尚无本轮实测，默认与发布状态未变。下一项GEMM-REL-03发布候选／独立验收协议冻结。封存独立验收、生产发布及无限GPU kernel恢复仍未通过；P1-05旧计时失败和P3-01候选A/A失败不变，不追加挑选重采。

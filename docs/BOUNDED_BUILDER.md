# 受限 CUDA builder 与源码验证

P3-02 在已声明的 RTX3070/SM86、固定 Docker image 下完成进程、文件、网络、资源上限及恢复验收。范围是普通容器／可信宿主控制边界，不是 GPU 驱动安全隔离、无限 GPU kernel 取消或生产发布。现有 dev2 wheel 继续提供已验证的 cuBLAS 开发运行入口；本页新增 builder 以仓库源码入口复现，未把它倒填为 dev2 wheel 功能。

## 复现

```text
python3 -m unittest discover -s tests -p test_build_artifacts.py
python3 -m experiments.p3_02.run_builder_3070 --evidence <new-dir>
python3 -m experiments.p3_02.audit_builder --evidence evidence/20261122-p3-02-bounded-builder --output <new-audit.json>
```

第二项通过既有私有 SSH 配置连接3070，只创建本次新目录／UUID容器，不改用户其他容器、镜像或原依赖目录。先复制已存在的公开pydantic依赖和Ninja到新目录，逐文件hash留档，不安装整套框架。project_root/deps_root是可信控制配置，不由模型提供；候选入口只接收三个有长度上限的CUDA文件bytes和本地reviewed SHA。模型密钥、参考、封存清单和发布状态不挂入builder或GPU worker。

## 构建与裁决

继续原KernelBench load_custom_model → load_trusted → Torch load_inline，原导出函数／SM86／cuBLAS链接／-lineinfo不变。KernelBench loader、dataset、timing及本地固定loader的hash均与原验收pin核对。仅CUDA源码变体另存，原文件不覆盖。

builder UID65534、network none、只读root／source／deps、CapEff0／NoNewPrivs1、无宿主可写mount；NVIDIA_VISIBLE_DEVICES=void且无--gpus，实际GPU不可用。PID128、CPU4、CPU内存8GiB、swap0，/build为1GiB tmpfs（允许native import）、/tmp512MiB、单文件32MiB、core dump禁用，20分钟外置期限。stdout仅有有界结果frame，编译输出重定向至stderr且最多1MiB；全部cache随容器清理。

结果只允许ZIP_STORED中四个固定常规文件：nka_gemm_v1.so、build.ninja、main.cpp、cuda.cu。包总上限9MiB，binary8MiB，其余分别限长；拒绝路径变更、重复、symlink、压缩／加密、大文件和错误ELF平台标签。可信宿主不导入新.so，只落盘、算hash，标记BUILT_UNVALIDATED。

随后由另一个受限GPU容器加载新.so，宿主事先对相同FP16量化输入计算CPU FP64参考。输出由宿主检查固定shape／FP32／finite与逐元素容差；成功只记录CORRECTNESS_ONLY。没有新计时／正式验收，就不晋升、注册或发布该binary。

## 本次实测

11个builder场景通过：三次fresh build成功；边界探测、stdout/stderr洪泛、8MiB测试临时区写满、无效返回、真实NVCC编译错误、native初始化abort及带child超时均隔离处理。边界探测／写满／无效返回没有合法artifact metadata，因而被REJECT；bad_archive在metadata检查即被拒绝，ZIP恶意路径／压缩／symlink等分支由11项独立CPU测试覆盖，不冒称同一个实卡fixture测到了后续ZIP分支。

27次GPU调用中，26次正确（含8个开发workload各cuBLAS／模板、故障恢复）；一份真实编译成功的零输出CUDA变体被独立FP64参考拒绝。超时后新builder重新编译成功，新GPU worker恢复正确。全部container已删，source／deps／控制best不变，实际P1冻结文件也未变。三次编译产物独立SHA，不假设可重复构建字节相同。

成功的生成main.cpp/cuda.cu/build.ninja与.so、失败stderr／源码、所有输入／输出及裁决保留。失败容器的tmpfs不另行导出，不能声称失败cache文件全量保留。传输按内容SHA去重、无损ZIP封装，本地相同内容通过只读hardlink归档；源字节不变。

[实测报告](../evidence/20261122-p3-02-bounded-builder/REPORT.md) · [CPU审计](../evidence/20261122-p3-02-bounded-builder/verification-pins.json) · [来源与取舍](../provenance/adaptations/20261122-p3-02-bounded-builder.md)

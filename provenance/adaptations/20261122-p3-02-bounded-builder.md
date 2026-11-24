# P3-02 有界builder与审查源码闭环（实施前）

2026-11-22，Asia/Shanghai。依据最新TASKS/HANDOFF，P1-05计时仍未验收，不重采；dev2安装／回退／回滚和runtime29项已留档，不重复整组。当前缺口是旧builder的宿主RW产物挂载、无界日志和新产物→独立GPU正确性联调。

| 固定来源／文件／符号 | 复用 | 具体缺口／改动 |
|---|---|---|
| KernelBench `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`, `eval.py::load_custom_model`；本项目已验收`gemm_regression::load_trusted` | 原loader、原Torch load_inline／导出函数／SM86／cuBLAS链接／-lineinfo，已有公开依赖与Ninja | 原load_inline会立即导入native模块，故编译与导入必须留在无GPU的受限builder；可信host绝不import新.so。保留原模块／CUDA，源码变体另存，不改受保护文件 |
| FlashInfer-Bench `15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee`, `TorchBuilder.build`、`Solution/SourceFile` | source身份／受控相对路径、Torch编译产物、固定entry、cleanup原则 | 原Builder本地cache／CUDA available判定和依赖不能直接代替无GPU安全边界；继续既有KernelBench路径，不安装整框架。仅3个CUDA文件可替换，loader／控制文件固定只读 |
| ForgeKernel `b2c8ae8510e02c28576808910a6b8827721ab7dc`, `harness_guard.py` | evaluator与合同固定、候选不能改控制／发布状态 | staged source/deps全只读；不挂项目／model key／reference／发布目录；before/after SHA核对 |
| KDA `acdd79343c5e9d0c7cb96289f26fd3f1051ce785`, `docs/agent-flow.md` | source→build→correctness→evidence／拒绝→保留best | build只得BUILT_UNVALIDATED；artifact自报成功不是正确性或晋升。错误源码／初始化退出／错误数值／host超时均留轨迹，随后新worker恢复；不写search best |
| 既有dev2 `bounded_io.capture`／`protocol`和Docker/NVIDIA runtime | 已验证字节／期限上限、先关attach再清理、unknown cleanup fail closed；既有GPU runner独立NPY输出 | builder改为无宿主RW bind，/build容量上限tmpfs（允许加载native）、/tmp临时区；明确NVIDIA_VISIBLE_DEVICES=void，无--gpus。stderr限1MiB，stdout固定ZIP_STORED产物包≤9MiB；closed-name parser限制文件数／单文件／总量／symlink／ZIP类型，host只写固定文件名，不extractall |

构建预定边界：UID65534、network none、read-only root、cap-drop ALL、no-new-privileges、log-driver none、sig-proxy false、PID128、CPU4、CPU memory/swap8GiB、MAX_JOBS1、/build1GiB tmpfs、/tmp512MiB、单文件32MiB、core=0；控制侧20分钟deadline。资源fixture可将/build降8MiB／timeout降2秒，不导入Torch。所有日志在启动前准备，builder stdout仅传header+flat metadata+ZIP；编译stdout重定向到stderr，原命令与生成main.cpp/cuda.cu/build.ninja随binary固定名回传。

预定命令：

```text
python3 -m unittest discover -s tests -p test_build_artifacts.py
python3 -m experiments.p3_02.run_builder_3070 --evidence <new-dir>
python3 -m experiments.p3_02.audit_builder --evidence <dir> --output <new-audit.json>
```

仅执行仓库已审查实现和明确的手写负例，不调用模型、不执行自由生成源码、不读封存清单。固定fixture：builder边界／stdout洪泛／stderr洪泛／tmpfs满／带child主机超时／伪造ZIP；真实编译错误、native初始化abort、可信基线fresh build与错误输出source fresh build；各失败后新GPU容器cuBLAS恢复，且至少一次失败后fresh builder成功。host先对FP16量化输入算CPU FP64参考，再在隔离GPU worker执行新binary、外置逐元素容差检查。无计时／profile／晋升／正式发布结论。

实测完成：11个builder场景／3次fresh build／27次GPU调用，26正确、零输出源码错误被外置FP64拒绝；loader4pin与实际P1freeze未变。CPU产物解析11项通过，归档／ZIP重解析／外置轨迹复算PASS。P3-02按固定image/SM86普通容器的OS进程文件网络资源与恢复范围验收；不含GPU驱动隔离、无限CUDA取消或生产发布。新增SSH只读进度查询因自动审批额度故障未执行，先前已授权回归正常完成并取回，未绕过。

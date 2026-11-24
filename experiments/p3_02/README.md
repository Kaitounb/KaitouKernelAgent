# P3-02 容器构建与恢复：阶段性入口

先复用固定 KernelBench `load_custom_model` → 原 `load_trusted` → Torch extension，保留已审查 CUDA／flags。FlashInfer-Bench 原 isolated runner 同 UID，且向子进程发送参考与计时；本项目只采用其 worker 生命周期，具体缺口及实施前来源见 [记录](../../provenance/adaptations/20261122-p3-02-worker.md)。不宣称完整 FlashInfer runner／Builder 集成。

现有固定 Docker image ID `sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a`、Python3.12.3、PyTorch2.14.0+cu130、CUDA13.0。脚本只创建 UUID 命名的新容器和 `/tmp/nka-p3-02-*` 工作目录；不改已有用户容器或镜像，不拉镜像。

```text
python3 -m experiments.p3_02.image_probe --evidence <new-dir>
python3 -m experiments.p3_02.worker_regression --evidence <new-dir>
python3 -m experiments.p3_02.build_regression --evidence <new-dir>
python3 -m experiments.p3_02.audit --evidence evidence/20261122-p3-02-runtime --output <new-audit.json>
```

前三项使用已授权3070与私有SSH配置，口令只进入子进程环境。最后一项仅做CPU证据／hash／生命周期复算。各GPU命令都使用新目录，失败不覆盖。依赖目标已有固定镜像、CUDA工具链和既有公开依赖；不是从空宿主重建环境的安装说明。

运行容器UID65534、network none、cap-drop ALL、no-new-privileges、只读根目录，只有固定源码／binary和输入只读挂载、单job输出目录可写；PID128、CPU2、CPU内存2GiB、tmpfs128MiB。可信宿主持有独立CPU FP64参考、裁决与发布状态，候选不能获得参考／模型凭据／独立验收清单。无pickle输出，读取输出禁止符号链接／FIFO并限制文件、header、shape和dtype。

15项是可信cuBLAS／模板各两shape、边界探测，加常数／输入污染／异常／符号链接／有限CUDA后主机等待超时五故障及每次新容器cuBLAS恢复。只验证正确性和隔离，不做性能结论。超时通过外置supervisor删除本次容器，不操作用户其他进程。

builder无GPU、无网络、UID65534、只读源码／依赖、CPU4／内存8GiB、PID128、MAX_JOBS1、20分钟外置超时；从空cache编译，产物写独立目录。既有公共依赖及Ninja复制到新副本并记录hash，原目录／镜像不变。编译后由独立runtime加载新binary，再跑15项和10项CPU输出解析故障测试。

当前范围限已审查源码和固定故障，不开放模型生成源码。GPU显存配额、恶意无限GPU kernel恢复、共享内核／驱动安全边界、输出／日志磁盘配额、通用未受信输出元数据解析与资源攻击验证、可迁移安装、独立验收、发布回滚仍需补齐。P3-02保持未勾选。

## 2026-11-22 实际结果

[原binary运行](../../evidence/20261122-p3-02-runtime/REPORT.md)15项PASS；[从空cache构建及新binary回归](../../evidence/20261122-p3-02-build-fixed-tool-path/REPORT.md)也通过15项，以及10项CPU输出解析检查。新产物SHA `66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3` 独立记录，原freeze不改。

```text
python3 -m experiments.p3_02.audit --evidence evidence/20261122-p3-02-build-fixed-tool-path --build --output <new-audit.json>
```

完整P3-02仍未勾选，下一步补通用输出／日志磁盘和资源攻击边界、可复现安装、实际版本回退／回滚。

## 后续有界通道与可安装开发包

[29项runtime实测](../../evidence/20261122-p3-02-bounded-runtime-confirmation-fix/REPORT.md)通过12故障／12恢复；[wheel新prefix安装](../../evidence/20261122-p3-02-install-rollback-dev2/REPORT.md)通过15次独立正确性、五场景回退和实际版本回滚。默认cuBLAS，development-only，未改变搜索best。source和raw前缀／输出／失败全保留。

```text
python3 -m unittest discover -s tests -p test_bounded_io.py
python3 -m unittest discover -s tests -p test_registry.py
python3 -m experiments.p3_02.run_bounded_3070 --evidence <new-dir>
python3 -m experiments.p3_02.install_regression --evidence <new-dir>
python3 -m experiments.p3_02.audit_bounded --evidence evidence/20261122-p3-02-bounded-runtime-confirmation-fix --output <new-audit.json>
python3 -m experiments.p3_02.audit_install --evidence evidence/20261122-p3-02-install-rollback --output <new-audit.json>
```

原image_probe/worker_regression/build_regression为历史阶段入口，新的kernel_agent/runner.py用于有界runtime。包及运行命令见[部署说明](../../docs/SM86_RUNTIME.md)。当前下一步是builder产物／日志有界化及受审查源码入口联调；完整P3-02／生成源码／正式发布仍未验收。

## P3-02收尾验收（2026-11-22）

[有界builder／审查源码闭环](../../evidence/20261122-p3-02-bounded-builder/REPORT.md)已完成11个builder场景／27个GPU调用，26正确、1个错误源码输出拒绝；3次fresh build成功、编译／native初始化／超时恢复及实际freeze不变通过。[组合验收](../../evidence/20261122-p3-02-bounded-builder/acceptance.json)。

```text
python3 -m unittest discover -s tests -p test_build_artifacts.py
python3 -m experiments.p3_02.run_builder_3070 --evidence <new-dir>
python3 -m experiments.p3_02.audit_builder --evidence evidence/20261122-p3-02-bounded-builder --output <new-audit.json>
```

旧“builder仍待”段落保留为阶段历史；当前P3-02已按固定image/SM86普通容器进程文件网络资源与恢复范围验收。无GPU驱动隔离／无限kernel取消／生产发布结论。新builder为仓库源码入口，不倒填为dev2 wheel功能。[说明](../../docs/BOUNDED_BUILDER.md)

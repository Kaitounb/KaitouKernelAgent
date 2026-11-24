# 3070 Attention / MLA：0.1.1 生产运行说明

2026-11-23。当前 RTX3070/SM86 已安装 **0.1.1**，`qualified` 通道仅允许固定可信 FlashInfer fa2 Attention 和经典 MLA core。后续运行不依赖4090，也不需要模型 API key。正式包和资格只属于下述已验收目标。

## 当前入口与直接运行示例

在3070的WSL中以可信控制用户root执行。目录、包和启动指针均保存在 `/home/dministrator/cuda-lab/nka-persistent`，不依赖重启会丢失的 `/tmp` 目录。

```sh
NKA_OPS=/home/dministrator/cuda-lab/nka-persistent/operators-0.1.1-xejiao7_
"$NKA_OPS/kernel-agent-operators" inspect --channel qualified --bundle "$NKA_OPS/sdk"
```

应返回 `status=qualified-runtime`、`production_release=true`、`scope=SM86_REVIEWED_FIXED_ATTENTION_MLA`。

已有两个可直接使用的公开合成示例。每次指定新的 `--work` 目录：

```sh
/home/dministrator/cuda-lab/.venv/bin/python -B "$NKA_OPS/tools/sm86_qualified_operators.py"   --operation attention --runtime-dir "$NKA_OPS" --work /tmp/nka-attention-demo-001

/home/dministrator/cuda-lab/.venv/bin/python -B "$NKA_OPS/tools/sm86_qualified_operators.py"   --operation mla --runtime-dir "$NKA_OPS" --work /tmp/nka-mla-demo-001
```

脚本生成FP16 NPY输入和request.json，并调用实际qualified CLI。Attention输出在 `outputs/output.npy`；MLA输出在 `outputs/z.npy` 和 `outputs/lse.npy`。执行轨迹在 `execution/execution.json`，包括版本、bundle、资源/日志界限、失败、清理与回退。示例文件已上传且两种请求均通过实际合同的CPU预检；GPU实现的独立/安装实测见下表。

使用已有数据时：

```sh
"$NKA_OPS/kernel-agent-operators" run --channel qualified --bundle "$NKA_OPS/sdk"   --request /absolute/path/request.json --inputs /absolute/path/inputs   --output /absolute/path/new-outputs --work /absolute/path/new-execution
```

`output` 和 `work` 必须是新目录，output的父目录须已存在。一次调用发布完整输出组；执行失败最多一次fresh worker恢复，清理未确认立即停止。损坏或非白名单SDK在GPU前拒绝。

## 实际验收

| 项目 | 证据 |
|---|---|
| 当前部署稳定性 | 完整三AA+三AB、1080原始Event段；AA最坏单对1.935923% < 原2%，最大组偏差0.447360% < 1.5%；AB最低保护0.999478866× ≥ 0.97 |
| 计时独立复核 | 1668文件SHA、80组确定性输入重建、1632输出数组NumPy FP64重算通过；Event合计1080.726秒 |
| Attention新独立验收 | rc1冻结后在3070可信控制目录生成全新合成清单并封存，8/8输出通过；239私有文件及NumPy FP64重算通过，最大绝对误差0.0001354902；模型只见hash和汇总 |
| MLA独立证据 | 同AOT/未变数值与合同的既有8调用/16输出独立验收继承，旧sealed未读取、传输或重跑；本轮公共输出另外做NumPy FP64重算 |
| 预览安装与恢复 | rc1新prefix/空cwd37CPU、13GPU输出/17尝试、两算子crash/timeout恢复、rc1→dev6→rc1实际包切换通过；2465文件审计 |
| 严格SDK准入 | rc2实际新包40CPU、2GPU输出、额外startup/bytecode/symlink和非对象manifest四类GPU前拒绝通过；4773文件审计 |
| 正式0.1.1安装 | 40CPU、9GPU输出/11尝试；7份qualified输出和2份显式退回rc2预览的输出，全部NumPy FP64复核；真实错驱动/篡改/范围及目录拒绝五项通过 |
| 正式包回滚 | 同一实际current指针0.1.1→rc2→0.1.1，切换后Attention/MLA分别实卡运行通过，最后恢复正式包 |
| 最终审计 | 5160文件SHA、全部NKA1/NPY/边界/回滚复核通过；12个运行文件与已实测rc2逐字相同 |

先前独立分配两侧输入的三AA最大5.598675%、组偏差2.162404%，保留拒绝、0AB。其decode单输入偏差随capture顺序翻转；现按原 `source_timing.paired_witness` 恢复双方共享同一物理只读输入ring，保留两Graph、1/16输入策略、前后hash/guards并检查device pointers相等。更正前冻结唯一一次方法更正批，原门槛不变，不证明全部旧抖动仅由分配造成。原540段不被覆盖或写回通过。

这里的性能是持续Graph批平均，排除计划/参考/H2D/控制和发布开销；不代表CLI单请求延迟、尾延迟或Agent优化收益。CUDA tensor最高分配486,093,824字节（约463.6MiB），不含context、缓存保留和桌面占用；8GiB显存适用于声明的合同/资源预算，无需清空日常显存。

## 支持合同和执行边界

- 目标：GPU UUID `GPU-859dd34e-59f1-959a-2546-ca027062f9bd`、驱动591.74、SM86；固定Docker image `sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a`，worker Torch2.14.0+cu130/CUDA13.0。其他GPU/驱动/image、改过的运行文件在执行前拒绝。稳定性实测使用当前HAGS On和普通时钟。
- Attention：连续BSHD FP16 q/k/v，batch1..16、1≤Sq≤Sk≤8192、heads1..128、D64/128，有限值绝对值≤4；可选全局bool `[Sq,Sk]` mask，causal为bottom-right；全mask行输出零。输出FP16。
- qualified MLA：FP16 absorbed q `[B,H,512]` / q_rope `[B,H,64]`，分页cache/rope_cache，int32 indptr/indices；B1..16、H16/32/64/128、L0..8192、**page16/split1**。页表必须为有效完整置换，有限值绝对值≤4。输出latent Z为FP16、自然LSE为FP32，空KV行Z=0/LSE=-inf。这是MLA core，不是完整模型层吞吐。
- 两算子输入总量上限192MiB。其他已审查page/split及Standalone Softmax保留development通道，不自动获得此资格。
- 只加载固定AOT bundle。完整目录集合须等于manifest；额外启动文件/字节码、symlink/special entry、文件变更均拒绝。SDK目录不添加缓存、日志或辅助脚本。
- 实测worker UID/GID65534、零capabilities、NoNewPrivs；network none、只读root和3份只读mount，控制/参考不挂载、凭据环境不透传。2GiB RAM、2CPU、32 PID、有界tmpfs/日志/输出和超时。GPU显存不是独立cgroup配额。
- 运行范围是固定可信SDK；不开放任意模型源码。超时演练是有限CUDA工作后的host等待，不声明GPU挂死恢复或驱动级隔离。原SM86 GEMM失败、旧独立数据和历史研究结论不被本次资格改写。

## 包回滚与恢复

当前实际指针：`/home/dministrator/cuda-lab/nka-persistent/operators-0.1.1-xejiao7_/current` → `/home/dministrator/cuda-lab/nka-persistent/operators-0.1.1-xejiao7_/prefix`。已验收的回退预览prefix：`/home/dministrator/cuda-lab/nka-persistent/operators-0.1.1rc2-o5m3a7aj/prefix`。维护时先停止新请求并确认没有项目任务，然后原子切换；以下两步不得并发执行：

```sh
# 回退到已验证的rc2预览（此时显式使用 --channel development）
ln -s /home/dministrator/cuda-lab/nka-persistent/operators-0.1.1rc2-o5m3a7aj/prefix "$NKA_OPS/next"
mv -Tf "$NKA_OPS/next" "$NKA_OPS/current"

# 恢复正式包（重新使用 --channel qualified）
ln -s "$NKA_OPS/prefix" "$NKA_OPS/next"
mv -Tf "$NKA_OPS/next" "$NKA_OPS/current"
```

预览回退会如实返回production=false；正式current已经恢复。原dev6/rc1/rc2、轮子、AOT、原始失败和私有封存数据保留。不要用新源码重建同版本wheel。

正式wheel：`artifacts/wheels/nvidia_kernel_agent-0.1.1-py3-none-any.whl`，SHA `f2ef0ec1f054a9331a4134e6545e6ce2a84b5aae22923f81063cb5c7418a2706`。AOT原zip SHA `939bf2d6d71a8d47fff652f944addb16857167e9f4ffbabd83c37a7e9a5ae5fa`，资格证书SHA `261b6fcb4c44ba3de7c88998b6d7e7f375350c3eb3f37b1d208be6eab0f9d3f4`。新prefix可用已有Python/NumPy和固定image离线安装；更换GPU/驱动/image不能继承当前证书。

[发行清单](../releases/sm86-operators-v0.1.1/manifest.json) · [正式审计](../evidence/20261123-sm86-operators-qualified-installed/verification.json) · [更正计时审计](../evidence/20261123-sm86-operator-deployment-shared-input/verification.json) · [Attention独立审计](../evidence/20261123-attention-sm86-independent/verification.json) · [MLA既有独立验收](MLA_RC1_ACCEPTANCE.md) · [示例源码](../examples/sm86_qualified_operators.py)。

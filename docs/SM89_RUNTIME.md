# SM89 固定可信 GEMM：0.1.0 生产通道

2026-11-23。`kernel-agent-sm89` 已在当前租赁 RTX4090 实例安装，`qualified` 通道只加载两份已验收 cuBLAS 产物。正式 wheel：`artifacts/wheels/nvidia_kernel_agent-0.1.0-py3-none-any.whl`，SHA256 `76224f3435764e5484e3434235e9f95a4b70cbe92476ac9f72c9df86bd3fb18f`。

## 已完成验收

| 项目 | 实际结果 |
|---|---|
| 稳定性与非退化 | 三AA+三AB，1728 Event段；单对最坏0.664015%、组偏差0.570454%，最低保护项0.999909889× |
| 公开正确性与Sanitizer | 320QA、32项Sanitizer，0 errors |
| 首次安装/预览回退 | 27CPU、9GPU输出/11尝试；crash/timeout、坏产物fallback、版本及rc2→新建dev7基线包→rc2切换均通过 |
| 新SM89独立验收 | 冻结rc2产物后在服务端生成并封存新合成清单，16/16输出通过；本机CPU FP64独立重算、246私有文件审计通过，最大绝对误差1.907349e-6 |
| 正式包安装 | 30CPU、6GPU输出/8尝试；其中5份qualified输出及1份显式退回rc2开发通道的输出，均正确 |
| 正式包回滚 | qualified内current↔baseline通过；0.1.0→rc2预览→0.1.0实际切换及恢复通过 |
| 最终审计 | 正式包581文件逐SHA、原始帧/NPY/实际边界及回滚复核通过；部署入口/新状态目录另有SHA增量快照 |

三类证据分开保存：性能实测、安装/恢复实测、新合成独立验收。旧3070/MLA封存清单未读取、传输或重跑。独立清单精确shape/seed和参考只在私有证据目录，模型只见commit和汇总。验收后没有修改数值执行代码：正式包的9个关键运行文件及两个CUDA二进制与已接受rc2逐字相同；新增内容是证据证书和qualified入口校验。

证书SHA `9451476fc9ac796ff1e5c4617c9414b0b7006366a0790f184618a8747a386c77`。它由真实稳定性、安装/回退、独立验收及CPU审计生成；没有使用模拟证书替代实卡验收。

## 当前已部署入口

以下命令在4090服务器上以可信控制用户root执行，路径指向已经安装的0.1.0。启动脚本及控制Python均在`/data`，不依赖原`/tmp`控制Python。

```sh
/data/nka-sm89-qualified-n2_3amub/kernel-agent-sm89 status \
  --root /data/nka-sm89-qualified-n2_3amub/state --channel qualified
```

实际状态为`qualified-runtime`、`production_release=true`，默认`sm89-cublas-current`，独立回退为`sm89-cublas-baseline`。

准备连续行优先FP16的`A[M,K]`、`B[K,N]` NPY文件后运行，输出为FP32 NPY：

```sh
/data/nka-sm89-qualified-n2_3amub/kernel-agent-sm89 run \
  --root /data/nka-sm89-qualified-n2_3amub/state --channel qualified \
  --worker-runtime /data/nka-sm89-qualified-n2_3amub/worker-runtime.json \
  --a /absolute/path/a.npy --b /absolute/path/b.npy \
  --output /absolute/path/c.npy
```

支持维度1..4096、有限输入且绝对值≤4，完整合同随包保存。每次运行写出版本、产物SHA、边界、资源/日志上限、清理结果和回退原因。活跃产物错误或执行失败时最多一次可信回退；清理未确认时停止，不继续调用GPU。

版本回滚：

```sh
/data/nka-sm89-qualified-n2_3amub/kernel-agent-sm89 rollback \
  --root /data/nka-sm89-qualified-n2_3amub/state --channel qualified
```

恢复current使用`activate --version sm89-cublas-current`并保留相同root/channel。回滚测试已执行；生产状态目录当前为current。旧dev6/rc1、dev7/rc2及其原始证据均保留，不用当前源码重建这些版本。

## 执行范围

- 只加载包内固定cuBLAS哈希白名单；SM89入口不接受`optimize`或任意源码/二进制路径。
- worker真实UID/GID65534，清除附加组和全部capabilities，NoNewPrivs=1。新IPv4/IPv6及其他非Unix socket被seccomp拒绝，CUDA所需本地Unix IPC保留。控制端环境不透传给worker。
- 产物/输入只读，私有控制/参考目录不可读，对控制进程发信号被拒绝；每次运行结束移除临时暴露目录。stdout/stderr、墙钟、CPU、文件大小/句柄及UID进程数有限制。
- 没有PID/mount/network namespace，也没有独立RAM cgroup；进程数rlimit按UID计。这里的qualified范围是固定可信SDK，不能作为任意模型源码/恶意CUDA的通用沙箱。源码生成的GPU执行继续使用已验证的SM86 Docker路径。
- 软件绑定为Python3.12.12、Torch2.7.1+cu126、CUDA12.6、RTX4090 UUID及driver570.86.10，以及已验收公开软件清单/解释器SHA。其他节点或驱动/软件变化在GPU前拒绝，先重新预检与资格绑定。维护时另建软件树，不在已接受软件目录内原地升级依赖。
- 性能证据是持续Graph批平均；不声明单请求尾延迟或优化提速。恢复测试为有限CUDA工作后的主机超时，不声明GPU挂死恢复。

NCU不是此运行通道的依赖。SM86、Attention/MLA的开发入口与状态保持各自验收范围，没有随本次SM89 GEMM被整体标为生产通过。

## 恢复安装与审计

正式包和依赖必须来自已核验归档；当前控制目录保留`tools/uv`及许可，可离线安装到新prefix。迁移时先核验软件/GPU绑定，不能直接继承当前节点证书。

[稳定性](SM89_CUBLAS_STABILITY.md) · [预览安装审计](../evidence/20261123-sm89-runtime-installed/verification.json) · [独立验收审计](../evidence/20261123-sm89-independent/verification.json) · [正式包审计](../evidence/20261123-sm89-qualified-installed/verification.json) · [部署增量](../evidence/20261123-sm89-qualified-installed/deployment-snapshot/verification.json)。

低权限机制使用现有[util-linux setpriv](https://man7.org/linux/man-pages/man1/setpriv.1.html)、Linux资源限制和[libseccomp参数规则](https://man7.org/linux/man-pages/man3/seccomp_rule_add.3.html)，没有增加宿主权限。

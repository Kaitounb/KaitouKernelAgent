# RTX3060 原生 Linux 对照

2026-11-24，按用户要求先在租用3060完成GEMM对照，再回3070验证。原3070普通动态7.699345%及100W/低温15.011395%的拒绝批次全部保留。

## 实际验收

| 项目 | 结果 |
|---|---|
| 目标 | RTX3060/12GiB/SM86，driver580.95.05，Ubuntu22.04 native host |
| 公开正确性 | 320输出通过，可信父进程FP64参考 |
| Sanitizer | 32输出，memcheck/synccheck零错误 |
| 故障恢复 | crash、有限CUDA完成后host timeout均拒绝，两次fresh恢复通过 |
| 稳定性 | 完整三AA，最坏单对0.479820%，最坏组偏差0.185669%，通过3%/1.5% |
| 保护项 | 完整三AB，最低0.999335845×，通过0.97 |
| 原始计时 | 1728段，1725.253秒CUDAEvent，全量保留，无筛选/重采 |
| 独立CPU审计 | 960文件逐SHA、原始帧/NPY/输入输出见证/校准公式/原gate重算PASS |
| 收尾 | GPU0%/1MiB/40°C，无compute进程/项目容器；170W未改 |

原始归档SHA：`487255f2f00b2bcaf5e92a35bfba6879e8bf88cfbfb3ecc8256614b800d6f0fc`。
协议SHA：`57948c94b2ed23ec58a0486ee1f023fdb84b47be86374f4bb18b41704ef690f9`。
完整结果见[审计记录](../evidence/20261124-sm86-3060-counterpart/verification.json)，复算命令：

```bash
/tmp/nka-delivery-verify-venv/bin/python -m experiments.full_delivery.audit_3060_counterpart evidence/20261124-sm86-3060-counterpart
```

## 运行边界与适配

固定KernelBench Graph/Event路径、两个已验证SM86 cuBLAS二进制原样复用，无新CUDA编译。Torch2.14.0+cu130/Python3.12.3/NumPy2.5.3运行于新image `sha256:7ed9d551ead159c3a4755f2a3ac84a01caf902bf3821495007828dc5eb18a4b7`。该新image需独立验收；五个核心数值库与3070原image实际加载SHA完全相同，见[runtime equivalence](../evidence/20261124-sm86-3060-counterpart/runtime-equivalence.json)。

3060 legacy GPU注入曾cuInit100/NVML Unknown，失败记录保留。按[NVIDIA CDI说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)使用已有nativeCDI device0/runc/visiblevoid解决，未修改宿主daemon、驱动或增加cap。实际GPU探针确认worker UID65534、零cap、NoNewPrivs、网络none、输入和产物只读、无Docker socket/主机私有目录/PID/凭据环境。可信父控制器单独持Docker socket，不将其权限算给worker。

GPU最高77°C，默认170W，未调功率或时钟。温度/时钟证据只描述实际环境，不将软件、驱动、OS、GPU同时变化的对照当成单变量根因证明。

## 结论范围

**3060固定可信cuBLAS的公开正确性、隔离、稳定性与非退化保护已通过。** A/B比较两份可信cuBLAS绑定，约1×符合负对照预期，没有优化收益或模型晋升结论。未新建独立封存验收、未签新生产证书、未切换部署入口；不能将该成绩回填3070。

随后3070执行一次固定8block NVML ON/OFF机制诊断，M1/16输入、1348固定replays、1200MHz，并同步记录Event与CPU/RAW耗时，结束恢复时钟。该小样本诊断不替代完整三AA/三AB。

原20GB慢传输partial、官方CUDA base拉取EOF、首个海外APT构建取消、legacy GPU注入失败均保留；来源和适配详[provenance](../provenance/adaptations/20261124-3060-gemm-counterpart.md)。本轮0新模型调用，旧sealed不读不传不跑。

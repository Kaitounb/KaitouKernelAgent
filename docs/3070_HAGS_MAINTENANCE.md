# 3070 硬件调度维护方案

2026-11-24 新方案尚未执行：3060对照通过而3070当前HAGS-On诊断仍不稳。已核对当前DWord2和原键值缺失备份，可用原Restore返回旧条件；没有改注册表或重启。若用户选择继续3070，需手动维护重启后DxDiag确认实际Off，再冻结全三AA→条件三AB；3%部署政策不变。原HAGS-Off2.040408%只是值得确认的线索。具体[计划](../evidence/20261124-sm86-hags-restore-plan/plan.json)、[诊断](SM86_NVML_DIAGNOSTIC.md)。下述2026-11-23内容是原启用方案与当时状态。

2026-11-23重启后覆盖：用户手动重启已完成；实际DxDiag HAGS已启用，bootUTC=2026-11-23T06:52:14.5000000Z。证据evidence/20261123-hags-after-user-reboot/runtime.json；不需要再次重启或Enable。新环境资格已单次启动，见evidence/20261123-cublas-hags-qualification。下面记录设置前事实与原维护方案，保留用于回退。

实际只读结果：Windows10 build19045，RTX3070驱动591.74/WDDM2.7，DxDiag明确 `Hardware Scheduling: Supported:True Enabled:False`。修改前注册表 `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\HwSchMode` 不存在，不能据不存在便猜测已开启。

[NVIDIA官方WSL2性能说明](https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements/)建议开启硬件加速GPU调度，以减少软件分包调度的开销。它是当前已确认的配置缺项；仍未证明它是所有旧计时离群的根因。

已执行并读回校验的可回退改动仅为：备份原值“缺失”及类型，写入 `HwSchMode` DWORD 2，读回校验。脚本 `experiments/full_delivery/windows_hags_setting.ps1` 默认Inspect；Enable必须使用尚不存在的备份路径，Restore按原备份恢复缺失或原值，不自动重启。

Windows需要重启才能应用该设置；这会中断当前Windows/WSL应用及SSH。当前SSH服务是Automatic，但没有确认FRP以自动服务运行，不能保证无人登录时自动恢复连接。建议先保存全部工作，由用户手动重启并恢复既有连接；若授权远程重启，需要明确应用已保存及维护窗口。不会强行关闭未保存应用，也不会创建额外驻留服务。

重启后验收顺序：

1. 确认已有SSH/FRP连接恢复，记录实际开机时间与驱动。
2. 持项目GPU锁，用DxDiag确认RTX3070实际Enabled:True，不能只看注册表2便宣称生效。
3. 冻结新环境身份及协议，继续原产物、8workload、双策略、三AA及条件三AB，2%/1.5%/.97门槛保持；旧2.040408%拒绝不覆盖。
4. 正式计时关闭Nsight；所有原始段、数值检查、环境、温度和恢复记录留档，只有实际结果达标才更新资格。

原数据已完整收回：锁频250ms三AA最坏2.849%，1秒持续平均三AA最坏2.040408%，均拒绝，无AB。两个新批次各320QA/864段，CPU审计与原始归档分列。没有生成资格证书，没有更改默认cuBLAS/回退或wheel。

完整DxDiag在本机 `.private/admission-repair-hardware/dxdiag-original.txt`（0600），公开状态与SHA在 `evidence/20261123-admission-repair/windows-dxdiag-scheduler.json`。4090 NCU另需平台配置，见[工单内容](4090_NCU_SUPPORT_REQUEST.md)。

设置收据：`evidence/20261123-hags-setting/result.json`。原值备份在 `C:\Windows\Temp\nka-hags-c52c669cfb9d4c52b454c0c390a707f1\backup.json`，本机同时保留原始备份对象；没有执行Windows重启。

设置后再次DxDiag核验仍为 `Enabled:False`，确认尚未生效；重启前Windows启动时间为 `2026-11-16T14:08:06.7925500Z`。证据 `evidence/20261123-hags-setting/runtime-before-reboot/runtime.json`。恢复连接后的只读检查命令：

```sh
/tmp/nka-p1-03-model-venv/bin/python -m experiments.full_delivery.check_hags_runtime \
  --output evidence/after-reboot-hags-runtime --require-enabled
```

该命令只查实际硬件模式；未启用则退出2，不启动任何资格任务。使用新的输出目录保留每次环境证据。只有实际启用后，另建带新环境身份的冻结资格；不要换output重跑已经拒绝的旧协议。

重启后验收结果：HAGS确已生效，但新身份320QA通过后仅完成AA0/288段，最坏5.142371%，随后空闲检查未满足而停止；474高负载样本1200/6800MHz、最高53℃，锁频复原成功。590文件及物理环境/原始输出CPU审计通过，qualification不通过，三AA未完成。HAGS不是已证明的旧离群根因，也不是充分修复；不继续重采该身份，未自动恢复HAGS原设置或再重启。见evidence/20261123-cublas-hags-qualification/verification.json。

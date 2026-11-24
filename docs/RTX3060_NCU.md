# RTX3060 NCU 实测与使用

2026-11-24：**真实NCU采集通过**。RTX3060/GA106、driver580.95.05，独立安装Nsight Compute CLI2025.3.1.0（官方包2025.3.1.4-1），已采集生产默认cuBLAS二进制的3次公开512×512×512 kernel，每次19项指标均有效。

## 实际结果

kernel为`Kernel2<cutlass_80_tensorop_s16816gemm_f16_128x128_32x4_nn_align8>`。指标包括SM活动、warp占用、Tensor Core活动、DRAM读写/吞吐、L2命中、三种stall、kernel时间和launch资源。固定KernelAgent原parser/schema与既有ncu_adapter解析通过；原`.ncu-rep`离线重新导出的CSV与首次采集数值完全一致。三份输出CPU FP64复核通过，最大绝对误差4.291534e-6。

- [CPU审计](../evidence/20261124-3060-ncu-smoke-v2/verification.json)
- [原始NCU报告](../evidence/20261124-3060-ncu-smoke-v2/segmented-archive/remote/cublas.ncu-rep)，SHA `94028848050a442a4050da9ff7acba4837950346509f517d3ad8575ab637a513`
- [原始CSV](../evidence/20261124-3060-ncu-smoke-v2/segmented-archive/remote/cublas.csv) · [解析后的全部指标](../evidence/20261124-3060-ncu-smoke-v2/parsed.json)
- [采集命令](../evidence/20261124-3060-ncu-smoke-v2/segmented-archive/remote/command.json) · [健康复查](../evidence/20261124-3060-ncu-final-postflight/result.json)

成功批38文件和首次失败批29文件逐SHA校验通过。首次容器在Python目标启动前因目录权限失败，实际0 kernel；随后只对可信诊断容器补DAC_OVERRIDE后成功。普通用户查询被驱动拒绝、旧版查询参数差异均有原始日志。无新CUDA编译/模型调用/正式Event采样，不读取私有验收清单。

## 查看现有报告（在3060 SSH内）

```bash
NCU=/home/featurize/nka-persistent/ncu-tools-5pb1aw2t/prefix/opt/nvidia/nsight-compute/2025.3.1/ncu
REPORT=/home/featurize/nka-persistent/gemm-release-ncu-igkbs15k/cublas.ncu-rep
"$NCU" --version
"$NCU" --import "$REPORT" --page details
```

离线读取报告无需sudo。新的真实采集需要管理员权限：驱动`RmProfilingAdminOnly=1`保持，不为所有用户开放计数器。本轮可信诊断容器UID0、SYS_ADMIN和DAC_OVERRIDE，仅挂载公开工作目录、固定artifact及工具，无Docker socket或凭据；生产worker继续UID65534/零cap。不要把任意模型生成源码放进此管理员诊断通道。

采集复用项目原profile_cuda：kernel replay、cache-control all、clock-control none、profile-from-start off、三次目标launch、有界超时。只作硬件诊断；其中的时间和占用不当作正式Event性能或提速结论，不重采已完成的1728段。

## 独立工具与恢复

[官方包元数据及安装回执](../evidence/20261124-3060-ncu-install/result.json)。服务器工具独立解包在`ncu-tools-5pb1aw2t/prefix`，未安装系统包或修改生产镜像。官方包SHA：`e2d393e1a0ba7ae1c983b1062f6494a8e85a98c987503f71bb73cc615aafe512`，314475378字节。

本机离线包：[nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb](../artifacts/tools/nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb)。[备份回执](../evidence/20261124-3060-ncu-install/local-tool-backup.json)记录整体SHA和截断下载的只读恢复。向同一已授权Linux服务器复制后，可在新工具目录中`dpkg-deb -x`解包；不要对生产镜像执行apt升级。已有GEMM恢复镜像与0.1.3 wheel保持原SHA。

原一次性安装/采集脚本已消耗，不重跑覆盖。未来新诊断须独立目录和公开输入；报告查看不需要再采集GPU。来源/取舍/命令见[适配记录](../provenance/adaptations/20261124-3060-ncu.md)，权限依据见[NVIDIA官方说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute)。

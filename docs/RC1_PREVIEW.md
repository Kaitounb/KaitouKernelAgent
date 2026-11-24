# 0.1.0rc1 未发布预览

该包完成新prefix/空cwd安装、19项CPU检查、5份真实GPU输出、crash/timeout恢复及应用包回滚。**生产资格未通过**：部署默认cuBLAS三A/A的最坏单对偏差4.274%超过2%，无A/B或生产证书；`--channel qualified`会在GPU前拒绝。开发运行结果仍明确 `production_release: false`。

wheel：`artifacts/wheels/nvidia_kernel_agent-0.1.0rc1-py3-none-any.whl`，SHA256 `d5b9ae9e1141b8b91bd937807918b11b5d46e467764c031115700e85bef6f7c4`。原dev6 wheel及其GPU证据保留，不覆盖同版本产物。

声明环境仍为已实测RTX3070/SM86、Ubuntu24.04 WSL2、驱动591.74、固定Docker image `sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a`。镜像内Torch2.14.0+cu130/CUDA13.0；控制进程Python≥3.12、NumPy≥1.26且<3。只使用既有镜像与公开依赖，不在运行时下载CUDA代码。

在该Linux控制环境中，以新prefix和新store安装：

```sh
NKA_WHEEL=/path/to/nvidia_kernel_agent-0.1.0rc1-py3-none-any.whl
NKA_PREFIX=/path/to/fresh-rc1-prefix
NKA_STORE=/path/to/fresh-rc1-store
python -m pip install --no-index --no-deps --target "$NKA_PREFIX" "$NKA_WHEEL"
export PYTHONPATH="$NKA_PREFIX"
"$NKA_PREFIX/bin/kernel-agent" init --root "$NKA_STORE"
"$NKA_PREFIX/bin/kernel-agent" status --root "$NKA_STORE"
"$NKA_PREFIX/bin/kernel-agent" run --root "$NKA_STORE" \
  --a /path/to/a.npy --b /path/to/b.npy --output /path/to/output.npy
```

从仓库外目录调用，确认导入路径来自新prefix。输入是FP16、连续二维NPY、有限值[-4,4]、M/N/K各1..4096；输出FP32。命令只选择已验收的cuBLAS二进制。错误或缺失的活动版本会选择可信回退；清理未确认则停止后续GPU尝试。语义和资源边界沿用[运行说明](SM86_RUNTIME.md)。

显式故障演练：`run`增加 `--inject-failure crash` 或 `--inject-failure timeout`，执行一次已审查故障后使用回退。版本回滚用 `kernel-agent rollback --root "$NKA_STORE"`。本次还实际将应用prefix从rc1切回归档dev6后输出正确；两种回滚的原始命令与数据在 `evidence/20261123-rc1-preview-installed/archive/remote/validation`。

新通道目前的预期行为：

```sh
"$NKA_PREFIX/bin/kernel-agent" init \
  --root /path/to/not-created-store --channel qualified
# 退出1：qualified GEMM release is not packaged
```

资格证书并未生成，CPU测试中的临时证书夹具不进入wheel。证书摘要、产物、合同、环境或准入状态不匹配都拒绝，不能用手改布尔值替代失败的实测。

`kernel-agent-operators`沿用原dev6的算子代码、catalog和AOT bundle；算子仍为开发范围。模型调用在本机管理端进行，密钥不传GPU机器；有界离线源码优化见[原入口说明](OFFLINE_OPTIMIZER.md)，正式研究的全局调用预算及防重发见[协议](FORMAL_STUDY_PROTOCOL.md)。

复核：

```sh
python -m experiments.full_delivery.audit_rc1_preview evidence/20261123-rc1-preview-installed
python -m experiments.full_delivery.audit_cublas_qualification evidence/20261123-cublas-final-qualification
```

前者返回安装/恢复审计PASS；后者返回 `AUDIT_PASS_QUALIFICATION_REJECT`，分别表示证据有效和资格失败。没有生产或优化收益结论。

MLA独立验收补充：冻结dev6默认16/1已一次性8/8调用、16/16输出通过，204私有文件CPU复核PASS。与RC1对应13个算子/帧/packet文件逐字相同，此为静态继承边界；没有另做一次RC1独立GPU验收。新状态及proof另存 `releases/full-project-rc1-mla-acceptance-update`；原预览manifest保持不变。[报告](MLA_RC1_ACCEPTANCE.md)。

2026-11-23修复补充：默认cuBLAS两个新增冻结资格各320QA/864段，锁频250ms2.849%、1秒持续平均2.040408%，仍拒绝，0AB/0资格证书。HAGS实际支持但关闭，现已备份配置DWORD2、待用户维护重启和实际生效验证；4090需平台增加计数器capability。旧包/二进制/封存验收未变，生产仍false。[修复证据](STABILITY_REPAIR.md) · [维护方案](3070_HAGS_MAINTENANCE.md) · [平台工单](4090_NCU_SUPPORT_REQUEST.md)。

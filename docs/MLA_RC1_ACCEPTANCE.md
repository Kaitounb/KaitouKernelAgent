# 冻结 MLA 默认实现独立正确性验收

2026-11-23，用户明确批准此次私有清单传输与执行后，按原冻结协议一次完成：4个独立workload×2个seed，共8/8 GPU调用、16/16输出正确。使用RTX3070/SM86、原dev6 wheel、许可完整AOT bundle，默认page16/split1。没有新增模型调用或性能采样，生产发布仍为false。

| 证据 | 实际结果 |
|---|---|
| GPU输出 | Z与自然LSE，最大绝对误差分别3.573324e-5、3.782725e-5 |
| 原冻结容差 | CPU FP64参考，atol=0.002、rtol=0.002，未放宽 |
| 独立CPU复核 | 204私有文件逐SHA、NKA1帧、NPY、16份输出与存档FP64参考结果全部通过 |
| 运行边界复核 | UID65534、network none、只读根与3份只读挂载、cap-drop ALL、no-new-privileges；input hash、offset guards、producer-consumer stream、protected hashes及清理记录通过 |
| 信息边界 | 参考在可信控制侧计算；候选只见本次请求和输入，不挂载完整清单、参考或模型凭据；模型未见精确shape/seed/NPY |

CPU审计复核存档FP64参考结果和绑定的可信evaluator源码，不声称独立重算整套MLA数学或另做GPU执行。该验收针对MLA core Z/LSE，不代表完整DeepSeek层吞吐或跨卡性能结论。

冻结及证据身份：

- candidate freeze：`a824c32ae625eb08cf27fe03f6359462344f0d5139f47a76eecfd26514123cea`。
- protocol：`e046fae01506f7681a1d7ab13e8142ff10496a34acdeaf8c2eede33e52302c4f`。
- dataset commitment：`d15407f8ead20dcc18ddd0e0bef2ee99628ff852860b63c4d635e264b602ae3b`。
- dev6 wheel：`d5f4bbf83dc050271af66baf35738f3800d02c2592b6fec101457122edc4fb38`。
- AOT bundle：`cfae38ce5c0fac56545207fe93588c1193646886fc5952f113fab487ef0be6eb`。
- 私有原始归档：`8f6f87b32182fce0f5cb954be591385e858e87b4c0eaee847f98facc42dd9806`。

公开汇总和CPU审计位于 `evidence/20261123-mla-independent-acceptance/{summary,verification}.json`。原始204文件、输入、参考、传输收据与reservation仅保存在本机私有目录及远端0700可信控制目录，不打入公开发布包。原自动审批拒绝保留于 `evidence/20261123-mla-private-approval-block`，后续具体授权另存 `evidence/20261123-mla-independent-approval/authorization.json`。

冻结验收使用dev6 wheel。RC1对应的11个算子文件与2个帧/packet依赖逐字相同，静态proof在 `evidence/20261123-mla-independent-approval/operator-wheel-identity.json`；没有为RC1重复一次独立GPU验收，也未重建任何wheel。旧预览manifest不改，新状态清单为 `releases/full-project-rc1-mla-acceptance-update`。

本次已执行命令：

```sh
/tmp/nka-p1-03-model-venv/bin/python -m experiments.full_delivery.run_sealed_mla
/tmp/nka-delivery-verify-venv/bin/python -m experiments.full_delivery.audit_sealed_mla
```

第一条已消耗一次性验收，不得重新执行或重建独立集；需要复核时仅运行第二条CPU审计，输出只有计数与hash。原GEMM独立16/16也不重跑。P6-04的独立正确性部分完成，最终性能资格仍未通过；另有4090 NCU权限阻塞，详见[生产准入](PRODUCTION_BLOCKERS.md)。

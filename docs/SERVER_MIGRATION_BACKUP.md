# 旧4090迁移备份已完成

2026-11-23，三个归档已收回本机并逐成员、整体SHA核验，合计约9.669GB。没有修改或释放旧实例，没有运行GPU/DeepSeek任务，也没有代替用户重启3070。

| 归档 | 压缩大小 | 常规文件 | SHA256 |
|---|---:|---:|---|
| project | 6.126GB | 25606 | `4dbbad9bc528d08a8c064e6dab0be168352d4956d4b7a517b0f862a8903fccb3` |
| critical | 0.01GB | 466 | `005ce2ca258c0c09b5ea322ae577f62a588e15556ca21234d6d61d8182f75e5e` |
| runtime | 3.532GB | 23808 | `91a6c44297b5753bb40d3211571013f46d124d91360a864f5e498ff6e915f831` |

私有备份目录：`/Users/kaitou/AI/NvidiaKernelAgent/.private/backups/20261123-server-switch-1fac00e5`。其中project为含私有状态的本机完整项目快照，server/critical为旧GPU代码/日志/原生库，server/runtime为完整venv、独立CPython和uv工具。1052个运行环境软链接目标均已归档；两份实测SM89原生库SHA一致。没有在Mac上执行远端Linux代码。

系统CUDA/NCU/驱动保存版本和路径，不是整机系统盘镜像。新服务器必须先实测Docker/真实NCU权限与软硬件兼容性；需要时按版本锁重建，旧性能结论不直接继承。原有独立验收和失败数据不重跑。

项目快照排除了备份目录自身、当前备份工作证据与临时缓存；备份完成后的文档/工作证据已补入私有latest-state增量并单独列SHA。本机project快照包含密钥和封存数据，只在本机恢复，不整包传到GPU。

恢复步骤与原始范围见 [恢复说明](../evidence/20261123-server-migration-backup/RESTORE.md)，完整证明见 [验证清单](../evidence/20261123-server-migration-backup/verification.json)。用户将自行重启3070并提供新服务器SSH，随后继续HAGS与跨卡准入验收。

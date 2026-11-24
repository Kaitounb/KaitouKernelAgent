# 来源与版本锁定

审查日期 2026-11-18。当前工程的五个主要来源、采用分工与状态见 [UPSTREAM_REUSE.md](UPSTREAM_REUSE.md)；固定版本见 [upstream-lock.json](upstream-lock.json)。已核对 35 个选定原文件，尚未集成或 GPU 运行。

旧本地材料审查见 [REUSE_AUDIT.md](REUSE_AUDIT.md)：Ascend 与旧教学材料目前保持 reference_only；P0-05 的 C++ 元数据／ABI 是保留原型，不是已确定的工程接口。完整更正见 [ROUTE_CORRECTION_20261118.md](../docs/ROUTE_CORRECTION_20261118.md)。

`upstream-lock.json` 与上游归档锁定五个主要项目，具体文件按其固定 commit／blob／SHA256 记录。`reference-lock.json` 另外锁定旧本地材料：

- references/ 的 61 个原文件，均匹配 MIGRATION.json；tree identity 由排序路径→SHA256 映射的规范 JSON 再计算 SHA256。
- P0-03/P0-04 的 21 个源文件和旧 evidence 的 3 个校验清单；清单内 107 个原始证据文件进一步校验。
- 两份补充许可／NOTICE，以及 GDN 固定 commit、原路径、来源名字差异与逐文件 Git blob 比对结果。

旧 optimizer 本地源码没有可核实 Git commit 或全包统一许可证；本次没有为它补造版本／授权。GDN 两文件的 Apache-2.0 头与 LICENSE 在固定本地 commit 核实，但并未采用到 CUDA 主工程。其余教学材料完整许可字段未知，使用状态仍为参考阅读；后续实际改编前需补齐。

许可文档从旧工程原样复制至 `licenses/`，未改写 reference 快照：

| 归档 | 适用范围 |
|---|---|
| `licenses/vllm-ascend-LICENSE` | 固定 vllm-ascend GDN 来源的 Apache-2.0 文本；与该 commit 的 LICENSE 一致 |
| `licenses/ascend-agent-NOTICE.md` | 原工程来源说明，保留旧 repo 名字及未随本快照迁移目录的描述；不是整个 NVIDIA 工程的许可声明 |

离线检查：

```bash
python3 tools/check_provenance.py
```

工具只读文件，不导入参考包、不连接 GPU、不改 lock，也不会打印私有文件内容。变更 reference 后应先保存新版本来源记录并审查采用关系，不以覆盖原始文件来通过检查。hash 提供内容身份和意外变化检查，本身不提供对同 UID 恶意篡改的安全隔离。

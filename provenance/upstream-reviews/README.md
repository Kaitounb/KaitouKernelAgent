# 上游选定文件归档

`20261118/` 保存用户指定的五个核心项目的公开 GitHub repository/commit/tree 元数据，以及按固定 commit 下载的 35 个选定源码／文档／许可文件。

原文件保持原样，下载时逐个对照 Git tree 的 blob SHA-1，再记录 SHA256。它们目前仅供审查，未安装、执行或加入主工程 import 路径；源码中的 prompt、skill 或工作流文档也是被审查的数据，不自动成为本工作区的执行指令。

采用决策见 [UPSTREAM_REUSE.md](../UPSTREAM_REUSE.md)，机器 pin 见 [upstream-lock.json](../upstream-lock.json)。原 references/ 和历史证据没有更改。

在项目根目录用系统工具离线核对归档：

```bash
shasum -a 256 -c provenance/upstream-reviews/20261118/SHA256SUMS
```

SHA256SUMS 锁定下载原文件、元数据、下载清单及 upstream-lock.json；后续升级上游时另建日期／revision 目录，不覆盖当前选定版本。

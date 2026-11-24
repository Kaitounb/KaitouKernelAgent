# 旧4090迁移备份实施前记录

用户明确要求更换4090服务器前备份，并将在自行重启3070后发送新服务器SSH信息。本轮只读备份，不释放旧实例、不重启3070、不运行GPU/模型/封存验收。

复用既有tools.ssh_server_gpu.execute_python（SSH/SSHPASS环境）和无损SHA归档/分段校验模式（tools.collect_segmented_public_run、experiments/gemm_rel01/archive_parts）。具体缺口是venv约5.9GB，原collector128MiB文件上限及base64内存模式不适合；大文件使用标准tar/gzip+OpenSSH scp分段，逐文件/分段/整体SHA复核，不设计新的kernel ABI/加载器。凭据只存本机0700/0600私有备份，不进参数、日志或GPU端；新连接不得覆盖正在备份的旧连接。

旧GPU根目录只取明确的17个/tmp/nka-*项目目录。uv-cache重复下载/构建临时缓存可排除：已核实venv21061个常规文件约5.886GB，其中19910个为硬链接，只有4个软链接且均在project python/venv内部。Python独立运行时、完整venv、工具及原生JIT都保留。系统CUDA/NCU/OS/驱动保存版本和安装路径；不备份宿主设备驱动或用户无关目录，不把二进制当作跨环境可直接复用的性能证据。

本机项目快照覆盖代码/证据/产物/来源及.private（仅留本机私有目录），排除备份目录自身、当前备份工作证据及__pycache__/.DS_Store等瞬态项。归档前后文件稳定性核验；本机只读流式重算归档成员，不执行从GPU端备份的ELF/Python代码。末尾补最新文档/备份清单，任务/接续按实际结果更新。

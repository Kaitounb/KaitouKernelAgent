# 用户修订部署A/A标准到3%

用户本轮提出将单对A/A噪声界限放宽到3%，并询问使用租赁4090验证稳定性。按此明确范围新增部署稳定性v2：单对偏差<3%，组几何均值偏离1仍<1.5%；完整三session、8开发workload、1/16输入策略、9对交替顺序不变。不修改搜索收益、正确性、.97保护项或冻结正式研究协议。

继续复用KernelBench commit423217d9fda91e0c2d67e4a43bf62f96f6d104f1的既有Graph/Event适配路径，以及KDA commitacdd79343c5e9d0c7cb96289f26fd3f1051ce785的外置证据/状态原则；实际数值直接调用kernel_agent.optimization.identity.group_log_ratio（来自本项目已冻结v1实现），不重新设计计时器。研究assess/identity、旧wheel与旧protocol均不改。缺口仅为用户新部署标准的版本化声明及CPU复核。

复核所有三份完整cuBLAS资格A/A和一份不完整HAGS批次，避免只展示最接近的一份；从逐文件SHA归档读取原始report，重算每组和每对，不采新GPU样本。明确标记修订发生在看过数据之后，历史复核不是预先冻结的新独立确认；旧原2%拒绝与原审计不覆写，A/A单项通过不能生成生产证书。最近HAGS批次三session缺失且已采5.142371%，仍不能通过新3%标准。

新4090实测使用同一v2门槛，实际环境、可信源码、SM89构建/二进制与执行边界须先核实并另行冻结，不继承SM86安装或性能资格。NCU可选。若无Docker/已验证的低权限隔离，只执行已审查可信SDK/固定cuBLAS测试，不跑自由模型源码。连接未收到，不租赁、不续费、不改旧凭据。

验证命令：/tmp/nka-delivery-verify-venv/bin/python -m experiments.full_delivery.reassess_deployment_stability_v2。只写新evidence，不重新运行旧GPU/独立验收或付费API。

实测：CPU_RETROSPECTIVE_REANALYSIS_COMPLETE，policy SHA8a56473b4ec221b27e88a9c21f96850b14118b74c2b47b889e6e25d3819dea59；四批逐文件归档核验，完整1秒最大组偏差0.2864655544%、最大单对2.040408%，v2单项通过；250ms2.849277%也通过，原始4.274482%及HAGS不完整5.142371%不通过。0新GPU/API、0生产证书。

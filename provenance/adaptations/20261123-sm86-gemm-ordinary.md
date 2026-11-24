# SM86 GEMM普通时钟部署资格（实施前）

继续固定KernelBench commit423217d9fda91e0c2d67e4a43bf62f96f6d104f1原load_custom_model/load_inline绑定及已验收cuBLAS二进制66fcc80f.../93fb2f43...；KDA acdd79343c5e9d0c7cb96289f26fd3f1051ce785与五上游pin、原GEMM freeze均不改。不新增CUDA kernel/ABI，不重编数值产物。

此前当前HAGS On只验证过1200MHz锁频环境（AA0/288段、最坏5.142371%，不完整且拒绝）。当前普通动态时钟的1.1部署环境尚无GEMM资格，Attention/MLA仅证明各自声明范围。本次新环境变化明确为普通时钟、用户已授权的GEMM部署v2（单对<3%、组<1.5%、三AA后才三AB、非退化>=.97），不是重启旧锁频失败或放宽正式研究/搜索门槛。

复用cublas_sustained_qualification_host.support_files和原有SourceRunner/两Graph共享输入ring、32调用/9对交替、1秒pilot公式；只去除clock control，加入普通时钟telemetry，并原样复用已通过SM89实测的sm89_cublas_host.assess_deployment纯数值门禁。保留2%研究assess为另列诊断，不混成3%部署资格。GPU执行仍在固定image低权限Docker中，可信CPU FP64参考不挂给worker。公开320QA与已有同二进制Sanitizer/独立验收来源分开记录，旧sealed不读/传/重跑。

先冻结新源/环境/policy/产物及审计代码；先public QA，连续3次utilization0才开始每轮计时；忙则停止未采部分，不清理系统/SSH/WSL进程，不自动重采。一次新普通时钟批，旧失败和所有原始样本保留，0新模型。通过后再做真实证据资格/新不可变版本安装、恢复、回滚；没有通过不生成生产标记。

生产绑定复用既有SM89 registry factory模式：只在GEMM规范CLI和其execute注入同一factory，复用原Registry与DockerRunner；新薄factory校验真实schema2证据、GPU UUID/driver/image、固定运行文件和v2 policy。原registry.py/deployment.py/runner.py/worker以及既有operator/SM89证书绑定文件不改，避免把其他已验收通道连带失效。新工作树0.1.2rc1；性能未通过不打包真实GEMM资格。

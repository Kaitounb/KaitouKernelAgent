# 实际两卡与专家路径范围

2026-11-23 最新：3070固定Attention/MLA 0.1.1 qualified已部署，三AA三AB及独立/安装/恢复/真实包回滚通过；不再依赖4090，详[运行说明](SM86_ATTENTION_MLA_RUNTIME.md)。下方历史跨卡研究不改。

已有资源由用户直接授权，未购卡、续费、申请新实例或扩大租赁。连接凭据仅在本机私有文件和SSH子进程环境。公开证据不含口令。

| 项目 | RTX3070 | RTX4090 |
|---|---|---|
| 计算能力/显存 | SM86/8GiB | SM89/24GiB |
| 系统 | Ubuntu24.04 WSL2 | Ubuntu22.04容器环境 |
| 驱动/CUDA | 591.74/13.0 | 570.86.10/12.6 |
| Python/Torch | 3.12.3/2.14.0+cu130 | 3.12.12/2.7.1+cu126 |
| FlashInfer | 0.2.8，仅两处C++17→20编译标志 | 原0.2.8 C++17 |
| MLA/Attention正确性 | 40公开+50开发+147配置与两Sanitizer通过 | 同覆盖通过 |
| 固定Attention/MLA生产运行 | **0.1.1 qualified**，当前GPU/driver/image绑定；MLA page16/split1，40CPU/9最终输出11尝试及5160文件审计 | 历史正确性/配置结果归档；非本次发布目标，不需继续租赁 |
| 固定GEMM生产运行 | 旧开发/独立证据保留，当前HAGS环境不继承SM89资格 | **0.1.0 qualified固定cuBLAS**：新独立16/16、实际安装/回退/包回滚及执行边界通过；不开放任意源码 |
| 自由源码worker | 固定Docker镜像的已测低权限边界 | 无Docker，user/net namespace EPERM，未验收 |
| NCU | 原生19字段/3launch；真实DSO hash和4.118秒成本 | ERR_NVGPUCTRPERM；缺宿主性能计数器能力 |
| MLA基线A/A | v2独占校准最坏3.184%，拒绝，不重采 | 三进程324段通过，最坏0.307% |
| 可信cuBLAS GEMM A/A v2 | 旧HAGS Off三AA2.040408%，按用户3%标准回顾复核单项通过；当前HAGS On不继承 | 新SM89环境三AA+三AB/1728段通过，最坏0.664015%、组偏差0.570454%，AB最低0.999909889×；917文件CPU审计通过 |

4090仅运行已审查的SDK及受控整数配置，不运行自由模型源码。NCU root权限不足失败如实保留，未修改云宿主配置或模拟硬件指标。目标卡重新优化的C硬件通道须等平台权限可用；普通正确性和配对计时可继续。两张卡软件版本不同，绝对延迟差不能证明agent收益。

经典FlashMLA专家对照固定为 `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`。官方支持矩阵为Dense Decoding/SM90/MQA/DeepSeek V3/V3.1；要求CUDA12.8以上。[固定版本README](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/README.md)。两卡均缺所需架构，因此不构建不运行，专家成绩为null。源码只作为兼容参考归档，SHA见 `provenance/expert-references/FlashMLA-ba89a3466e94/record.json`，未声称集成或性能。

实际可运行的成熟对照是固定FlashInfer fa2（包括SDK自己的内部split planner）；显式split/combine方案与此默认配置比较。它不能替代FlashMLA的名称或成绩。MLA核心边界为已吸收query/cache→Z和自然LSE；投影、RoPE、append、metadata、cache重排另列。

目标卡普通MLA调参已补齐：page16/split2、page64/split1、page1/split1各三AA+三AB共5184段，全部独立审计通过、候选均拒绝。默认16/1保留；目标C仍未运行。[逐项结果与限制](CROSSCARD_RESULTS.md)。

2026-11-23修复补充：默认cuBLAS两个新增冻结资格各320QA/864段，锁频250ms2.849%、1秒持续平均2.040408%，仍拒绝，0AB/0资格证书。HAGS实际支持但关闭，现已备份配置DWORD2、待用户维护重启和实际生效验证；4090需平台增加计数器capability。旧包/二进制/封存验收未变，生产仍false。[修复证据](STABILITY_REPAIR.md) · [维护方案](3070_HAGS_MAINTENANCE.md) · [平台工单](4090_NCU_SUPPORT_REQUEST.md)。

新SM89可信cuBLAS完整实测与范围见[报告](SM89_CUBLAS_STABILITY.md)。本轮无新NCU采集，固定源码native执行边界已明示，不将其写成Docker或自由源码worker验收。

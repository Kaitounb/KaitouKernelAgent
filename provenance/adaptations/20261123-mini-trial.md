# 小规模36提案接入前记录

复用固定KernelAgent provider（e0647170...，utils/providers/openai_base.py）及已有CallBudget/prepare_channel；复用Git patch guard、安装dev5有界source pipeline、KDA式proposal→review/build/QA/paired decision记录。KernelBench的CUDA构建/校验/Event及FlashInfer0.2.8 SDK已分别在GEMM/MLA绑定，不新增kernel/loader。原frozen bundle45及GEMM协议不变。

必要缺口是2任务×3方法×6顺序提案的状态/历史与精确身份缓存：新增mini_trial.py只编排/生成可审查提案，不直接执行自由源码。新源码必须由可信控制侧审查SHA后进入既有Docker路径。MLA只允许已审查SDK整数配置；两方法相同空间，A仅常规page参数。GEMM A仅warps，B/C是同一固定CUDA device body差分。使用原prepare_channel白名单，B不调用profile loader，C的原始CSV/真实DSO及wall成本有绑定。源码修改范围与每任务固定父版本写进冻结协议，不从其他方法传递best/提案；准确重复可复用已审计结果但仍记预算。

每个分支必须先落盘上一提案的工具结果/裁决，才允许下一提案；未知调用不重试。两任务共36attempts，A不调用模型，B/C最多24新API且受700全局账本限制。费用没有账单接口则null，保留实际token/API wall/编译/验证/原始Event/profile成本，不把null当0。校准失败和无稳定收益结果照实保留；研究状态与生产激活分离。先小规模实际成本验收，后冻结正式协议，不能把本批或前期pilot叫正式消融。

验证：freeze source snapshots和protocol；dry prepare无API；同起点B/C common_sha/B0/C1检查；首提案真实API与已有身份查重/人工语义审查；全部独立工具结果再外置审计36条账本，任何未完成不勾P3-04。

A控制的warp1是新的未采样参数身份，复用旧gemm_candidate::{checks,paired}和93固定binary、28项前后QA、32Graph×12replay，不重采4/8/K2/B拒绝。原三份KernelBench eval/dataset/timing逐字复制，已检查宿主requests有但pydantic缺失，复用现有 /tmp/nka-gemm-rel01-f3ekx0tu/run/deps 中公开pydantic/core，不新增安装。新run_gemm_parameter包装仅分离AA→条件AB并加同卡flock/空闲入场/证据记录。

源码执行adapter复用原run_feedback_evaluation的dev5命令和lossless-sha-segments-v1传输；只增加proposal-dir/review-file参数、当前controller源码快照和同卡flock。旧默认B路径与内层builder/worker/QA/paired gate不改。改前脚本已存mini-v2/sources-before-adapter，新源码239802...的A/B指针局部变量提取经可信语义审查，两个方法完全同源只执行一次。

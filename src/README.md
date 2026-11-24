# 本地适配与保留原型

当前只有 P0-05 的 `contracts.cpp`：本地主机元数据／跨度检查，已做 C/C++ 检查；没有 GPU 运算或成熟 agent 集成。它作为教学／设计原型保留，不作为默认工程合同。

后续先使用五个主要上游的构建、评测、控制和 profile 组件，再在这里放确需的任务／硬件连接代码。成熟 Python 可以保留，不计划按自定模块清单重写完整框架。[采用表](../provenance/UPSTREAM_REUSE.md)、[接入顺序](../docs/CUDA_ENGINEERING.md)。

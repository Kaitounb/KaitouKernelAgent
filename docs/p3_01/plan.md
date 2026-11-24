# P3-01 首轮可执行计划

1. 复用experiments/p1_03/propose.py::provider_class，读取已授权本机私有DeepSeek配置；不复制密钥到远端。使用已准备request，一次调用，保留请求、原响应、finish_reason、tokens／墙钟和失败，禁止自动重试。
2. 严格JSON／parent绑定；调用experiments.p3_01.patches在临时目录用git apply --no-index校验单CUDA文件补丁。禁止路径／mode／rename／binary及license／host launcher修改，不能改评测／合同。由可信控制侧审查实际CUDA改动，不要求用户重复授权。
3. 新source identity、父版本和源码另存；通过已验收DockerBuilder无GPU构建，artifact仍为BUILT_UNVALIDATED。真实候选增加K=16/80等尾部、多seed、stream／输入不可变／边界与需要的Sanitizer检查；P3-02的数值fixture不代替完整新候选合同检查。
4. 正确后复用原KernelBench／配对Event路径，适配双binary基线与新源码候选，先为新identity验证A/A再采普通Event；不改旧4-warps的P1-05失败或原晋升阈值，不把无profile提案当B/C结论。
5. 外置裁决与回放决定是否保留；无新稳定计时和正式验收不得晋升／发布，生产默认cuBLAS开发包不自动切到新source。

上述为实施前计划。2026-11-22审批服务恢复后真实调用一次、2,232tokens；hunk计数失败保留，Git --recount派生后审查，新build／160项正确性／Sanitizer各8项0errors。新源码一批3session／864段A/A最坏3.233%超界，REJECT，无A/B／晋升／收益。本机完整证据审计PASS，P3-01按首轮受审查源码及新证据拒绝范围验收，详见[实测](../P3_01_SOURCE.md)。不新增API调用预算，原P1-05计时失败不变。

已使用的单次发送入口（预算不因本轮完成而重置）：

```text
python -m experiments.p3_01.propose --prepared evidence/20261122-p3-01-source-prepared --output <new-call-dir>
```

4项mocked provider测试已验证一次调用／异常不重试／不完整输出拒绝／预算漂移先拒绝，并明确标记非真实API；key不落盘、不传远端。

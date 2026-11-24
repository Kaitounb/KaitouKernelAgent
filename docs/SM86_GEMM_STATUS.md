# 3070 GEMM 最终资格状态

2026-11-24 最新范围：RTX3060 GEMM 0.1.3正式交付已完成，完整三AA三AB/独立16项/真实回退/入口和恢复备份通过。[3060正式运行](RTX3060_GEMM_RUNTIME.md)。本次无新增阻塞；3070 GEMM原拒绝及4090目标C延期仍保留，3070 Attention/MLA0.1.1保持。下方为历史或其他目标，不能据其旧production=false覆盖本发行。


2026-11-24：GEMM 生产性能资格仍未通过。已交付并实测的是0.1.2rc1开发预览；原0.1.1固定Attention/MLA生产范围保持。

| 3070条件 | 公开QA | 三AA Event段 | 最大单对偏差 | 最大组偏差 | 部署v2 |
|---|---:|---:|---:|---:|---|
| HAGS On/普通动态频率 | 320 | 864 | 7.699345% | 0.889524% | 拒绝 |
| HAGS On/100W动态频率 | 320 | 864 | 15.011395% | 1.387542% | 拒绝 |

门槛保持单对严格<3%、组偏差<1.5%；两批都0AB/0证书，不筛异常或覆盖原拒绝。615/616原始文件逐SHA、NKA1/NPY、FP64回执、Graph见证及统计CPU审计通过，表示证据一致，不表示性能达标。

普通批最高88°C，目标温度83°C；一次100W/90秒热诊断最高54°C、热降频计数无增加、恢复240W后，才预先冻结唯一100W资格批。100W完整批最高55°C、热计数仍无增加却出现15.011395%偏差，因此高温不是全部抖动的充分解释。100W是[NVIDIA官方power-limit](https://docs.nvidia.com/deploy/nvidia-smi/)报告的最低允许值；无功耗搜索。

追加native/100W小M时间线诊断（带profiler、不同于Docker正式样本）54段/73040次Graph活动CPU重放通过。最坏诊断配对中Graph总执行时间约971.7对935.4ms，而Graph之间空隙总差约0.63ms；支持继续调查Graph执行期间的变化，不能确定正式15%异常的具体根因。profiling开销、不同CPU限制与WSL时间戳换算精度限制均保留；首次捕获前遗漏cuBLAS初始化的0计时段失败trace也完整保留。

0.1.2rc1实卡安装：49CPU、11GPU输出13尝试；GEMM crash/有限CUDA后host timeout/坏产物回退、版本及rc1→0.1.1→rc1实际包切换、两既有算子qualified功能回归和直接容器边界均通过。592文件与全部帧/NPY/NumPy FP64独立复核PASS，wheel SHA256 `1f196c84e93ad406ddef58f55118041a43fa04538bcd9f15fa837d6fbf341264`。GEMM qualified缺证书在GPU前拒绝；新GPU/driver/image/运行代码/policy及100W温度门禁只是候选实现，未签发证书或GA。

收尾已实查GPU0%/40°C、240W恢复，无项目容器；旧Attention/MLA真实qualified inspect仍通过。没有新增模型调用、CUDA编译或旧sealed执行，HAGS/驱动/原部署入口未改。

3060对照现已完成：320QA/32Sanitizer/两fault恢复、三AA三AB1728段、960文件CPU审计PASS，最坏0.479820%/组0.185669%、最低保护0.999335845×。3060五核心数值库与3070原image逐SHA相同；原生OS/驱动/GPU同时不同，不能确定具体根因，也不能将3060成绩回填3070。详[完整对照](SM86_GEMM_3060_COUNTERPART.md)。

随后回3070固定8block诊断144段/141文件审计PASS，ON最坏17.024809%、OFF仍5.272610%，Event与RAW耗时重现波动，关监测不足。时钟已恢复、240W保持，0新证书。详[诊断及后续方案](SM86_NVML_DIAGNOSTIC.md)。

下一步恢复原HAGS状态的可回退方案已经只读核对，未改注册表/重启；需要用户选择维护窗口，Windows重启会中断应用/WSL/SSH。原HAGS-Off2.040408%只作后续确认线索，不预称充分修复。也可选择在3060独立完成GEMM新验收/安装/回滚发布。当前两卡均无项目GPU/API在途。

[普通批审计](../evidence/20261123-sm86-gemm-ordinary-qualification/verification.json) · [100W审计](../evidence/20261124-sm86-gemm-power-qualification/verification.json) · [时间线审计](../evidence/20261124-sm86-gemm-power-timeline/replay-verification.json) · [预览运行](SM86_GEMM_PREVIEW.md) · [3060预检](../evidence/20261124-sm86-3060-preflight/preflight.json)。

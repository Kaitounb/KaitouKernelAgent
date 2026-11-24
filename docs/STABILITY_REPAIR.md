# SM86 最终计时稳定性修复

2026-11-23，当前状态：1秒持续平均协议已完整执行并独立CPU审计，最坏2.040408%仍超2%，拒绝。已配置HAGS DWORD2，等待Windows维护重启；实际运行HAGS尚未确认开启，不得标为达标。所有旧失败保留；无新模型调用、无封存数据重采、默认与回退二进制未变。

## 已证实的结果

| 检查 | 实际结果 |
|---|---|
| 原默认cuBLAS部署资格 | 320QA正确；三AA/864段最坏4.274%，拒绝 |
| WSL锁频接口 | 官方WSL nvidia-smi返回Unknown Error |
| Windows原生管理员锁频 | 可设置与恢复；负载实测SM1200MHz、显存请求7001MHz/实际CUDA P2约6800MHz |
| 锁频后250ms窗口一次资格 | 320QA正确；三AA/864段最坏2.849%，仍拒绝；没有AB |
| 锁频实际遥测 | 452个高负载样本严格1200/6800MHz，最高54℃；退出恢复通过 |
| 锁频后1秒窗口一次资格 | 320QA正确；三AA/864段最坏2.040408%，拒绝，无AB；614文件审计通过 |
| 1秒批次遥测 | 1585高负载样本1200/6800MHz，最高53℃，退出恢复通过 |
| CPU cgroup观察 | 10秒观察窗口nr_throttled=0、throttled_usec=0；不能归因于已证明的CPU配额节流 |
| 第一份Nsight trace | 162 NVTX与CPU API存在，GPU活动缺失；仅保留，不作为完整时间线 |
| 官方WSL时间戳修复后trace | 78438 GPU Graph活动，全部162范围与runtime correlationId及replays数量对应；一个诊断pair相差5.52ms |

修复后诊断的该配对CPU提交均约4ms，慢侧最长Graph约1.728ms，对应普通中位约0.585ms；全部诊断的最大跨Graph空隙约0.282ms。现有证据指向短时设备执行变化，不能断言具体WDDM调度、外部进程或功耗原因。Nsight官方WSL时间戳转换修复有精度限制，诊断不作为最终性能资格。

## 新协议及范围

原始指标是每段32call Graph反复执行所得的批量平均。新协议将目标段从250ms固定为1000ms，以降低已观察到的毫秒级短时变化对平均值精度的影响。不是单请求尾延迟保证，也不证明250ms旧协议通过。

保留全部8个公开开发workload、1/16输入两策略、每组9对、三独立进程AA及条件三独立进程AB，保留2%单对偏差/1.5%组均偏差/.97部署保护门槛，保留所有原始段。不删除、裁剪、选择离群，也不按最终数据再挑选窗口。

唯一计时参数变化：4-replay pilot的三次中位数，replays=`clamp(ceil(4000/median_pilot_ms),1,8192)`，每段仍为32×replays次调用。非Sanitizer worker timeout180→600秒以容纳16行的长窗口，其余容器资源/低权限/只读/无网络/CPU配额不变。Windows原生锁频1200/7001请求与全程遥测、finally恢复沿用已验证版本。

协议与源码在 `evidence/20261123-cublas-sustained-qualification` 执行前冻结；只能执行一次该身份。CPU support delta确认只有超时、窗口/pilot数值变化，原CUDA源码、数值参考、候选和input/output witness逻辑不变。通过后也只能声明这一物理环境与持续平均指标的验收结果。

## 证据入口

- `evidence/20261123-admission-repair`：两卡环境、原生锁频探针、CPU配额观察、CPU gate/delta检查及原失败分析。
- `evidence/20261123-cublas-locked-qualification`：320QA/864段真实锁频250ms拒绝、613文件归档及CPU审计。
- `evidence/20261123-cublas-timeline-diagnostic`：首份不完整Nsight原始报告。
- `evidence/20261123-cublas-timeline-timestamp-fixed`：官方配置修复后smoke/完整trace/SQLite/逐范围关联与原始Event。
- `evidence/20261123-cublas-sustained-qualification`：已结束且拒绝的持续平均资格。

NCU权限是另一项平台限制，见[4090工单内容](4090_NCU_SUPPORT_REQUEST.md)。没有代发工单或更改4090宿主权限。

HAGS后续：DxDiag实际确认支持但关闭，而不是依据注册表缺失推断；已按官方建议配置并备份，尚未重启。维护与回退见[方案](3070_HAGS_MAINTENANCE.md)。不继续延长窗口挑通过；实际HAGS生效属于新的物理调度环境，必须先验证再冻结新资格。两批新增成本见 `evidence/20261123-admission-repair/costs.json`：640QA、1728原始段、Event合计1081.675秒（不含QA/热身等），0新DeepSeek。

只读复核命令：

```sh
/tmp/nka-delivery-verify-venv/bin/python -m experiments.full_delivery.audit_cublas_locked_qualification evidence/20261123-cublas-locked-qualification
/tmp/nka-delivery-verify-venv/bin/python -m experiments.full_delivery.audit_cublas_sustained_qualification evidence/20261123-cublas-sustained-qualification
python3 -m experiments.full_delivery.audit_cublas_timeline evidence/20261123-cublas-timeline-timestamp-fixed
```

前两条的审计PASS表示数据与拒绝裁决可复算，不表示资格通过。计时源码/协议已经消耗一次，不得通过修改output重跑。

## 用户重启后的HAGS新环境实测

2026-11-23：实际HAGS Enabled True，新bootUTC=2026-11-23T06:52:14.5000000Z。先按归档SHA恢复持久dev6/AOT，无重建/独立验收；14个有限活动高负载样本验证锁频1200/6800MHz。新协议ec6bbbc...沿用原字节sustained host及所有门槛。首次脱离SSH的后台启动在原生锁频前发生UtilAcceptVsock超时，0QA/0Event，21文件保留；保持前台SSH父会话后仅继续未采样阶段，协议/host/二进制不变。

320QA通过、AA0共288原始Event段完成；dev03/ring16=4.731864%、dev06/ring16=5.142371%、dev07/single=3.464863%、dev08/ring16=3.274812%，均超2%。下一轮前10次空闲观察出现0%与1/3/5%间歇活动，未满足连续3个0%，状态GPU_BUSY_UNSAMPLED_REMAINDER。完整3AA未完成、0AB；474高负载样本均1200/6800MHz，最高53℃，finally两reset成功。已采超限不能通过补跑或筛选消除，不续采该身份。

590原始文件归档SHA5fee025b048fc64c082f94613bdce819f400e70c65ea7950fc62bfa6b65102a8；物理身份、全文件SHA、NKA1/NPY/参考receipt/Graph/pilot/时钟CPU审计AUDIT_PASS_INCOMPLETE_QUALIFICATION_REJECT。partial审计不改变原assess，明确三AA缺失且无资格。新实测不能证明HAGS改善或恶化的因果关系；后台活动来源与运行期干扰尚未定位。0新增模型、无证书/发布，无sealed执行，旧拒绝保持。

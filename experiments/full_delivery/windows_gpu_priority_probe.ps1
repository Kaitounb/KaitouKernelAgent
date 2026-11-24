[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class NkaGpuPriorityQuery {
 [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint rights, bool inherit, int processId);
 [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
 [DllImport("gdi32.dll")] public static extern int D3DKMTGetProcessSchedulingPriorityClass(IntPtr process, out int priority);
}
'@
$engines=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Where-Object UtilizationPercentage -gt 0 | Select-Object Name,UtilizationPercentage)
$processIds=@($engines | ForEach-Object { if ($_.Name -match '^pid_(\d+)_') { [int]$matches[1] } })
$processIds+=@(Get-Process -Name 'vmmem*','vmwp*','wslhost*','wslservice*' -ErrorAction SilentlyContinue | ForEach-Object Id)
$processes=@()
foreach ($gpuProcessId in @($processIds | Sort-Object -Unique)) {
 $gpuProcess=Get-Process -Id $gpuProcessId -ErrorAction SilentlyContinue
 if ($null -eq $gpuProcess) { continue }
 $handle=[NkaGpuPriorityQuery]::OpenProcess(0x1400,$false,$gpuProcessId)
 $priority=-1;$returnCode=$null;$lastError=0
 if ($handle -ne [IntPtr]::Zero) {
  try { $returnCode=[NkaGpuPriorityQuery]::D3DKMTGetProcessSchedulingPriorityClass($handle,[ref]$priority) } finally { [void][NkaGpuPriorityQuery]::CloseHandle($handle) }
 } else { $lastError=[Runtime.InteropServices.Marshal]::GetLastWin32Error() }
 $processes+=@{Id=$gpuProcessId;Name=$gpuProcess.ProcessName;GpuPriority=$priority;QueryReturnCode=$returnCode;OpenProcessError=$lastError}
}
@{Scope='Read-only GPU engine owners and official D3DKMT GPU scheduling class';Engines=$engines;Processes=$processes} | ConvertTo-Json -Depth 5 -Compress

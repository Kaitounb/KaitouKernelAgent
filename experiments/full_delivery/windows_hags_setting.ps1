param(
 [ValidateSet('Inspect','Enable','Restore')][string]$Action='Inspect',
 [string]$BackupPath
)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$keyPath='HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
function Read-Setting {
 $key=Get-Item -LiteralPath $keyPath
 $exists=@($key.GetValueNames()) -contains 'HwSchMode'
 @{Exists=$exists;Value=if($exists){$key.GetValue('HwSchMode')}else{$null};Kind=if($exists){$key.GetValueKind('HwSchMode').ToString()}else{$null}}
}
$before=Read-Setting
if($Action -eq 'Enable') {
 if([string]::IsNullOrWhiteSpace($BackupPath)){throw 'BackupPath required'}
 if(Test-Path -LiteralPath $BackupPath){throw 'Existing backup: do not overwrite or replay enable'}
 $parent=Split-Path -Parent $BackupPath
 if(!(Test-Path -LiteralPath $parent)){[void](New-Item -ItemType Directory -Path $parent)}
 @{KeyPath=$keyPath;ValueName='HwSchMode';Original=$before;Action='Enable';RequestedValue=2;RebootPerformed=$false} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $BackupPath -Encoding UTF8
 [void](New-ItemProperty -LiteralPath $keyPath -Name 'HwSchMode' -PropertyType DWord -Value 2 -Force)
} elseif($Action -eq 'Restore') {
 if([string]::IsNullOrWhiteSpace($BackupPath)){throw 'BackupPath required'}
 $saved=Get-Content -LiteralPath $BackupPath -Raw | ConvertFrom-Json
 if($saved.KeyPath -ne $keyPath -or $saved.ValueName -ne 'HwSchMode'){throw 'Backup scope mismatch'}
 if($before.Exists -and $before.Value -ne 2){throw 'Setting changed by another actor; do not overwrite'}
 if($saved.Original.Exists){[void](New-ItemProperty -LiteralPath $keyPath -Name 'HwSchMode' -PropertyType $saved.Original.Kind -Value $saved.Original.Value -Force)}
 else {Remove-ItemProperty -LiteralPath $keyPath -Name 'HwSchMode' -ErrorAction SilentlyContinue}
}
$after=Read-Setting
if($Action -eq 'Restore') {
 if($saved.Original.Exists) { if(!$after.Exists -or $after.Value -ne $saved.Original.Value -or $after.Kind -ne $saved.Original.Kind){throw 'Restore verification failed'} }
 elseif($after.Exists){throw 'Restore of absent value failed'}
}
if($Action -eq 'Enable' -and (!$after.Exists -or $after.Value -ne 2 -or $after.Kind -ne 'DWord')){throw 'Post-write verification failed'}
@{Status='PASS';Action=$Action;Before=$before;After=$after;BackupPath=$BackupPath;RebootPerformed=$false;ActualRuntimeHagsState='Not inferred from registry; verify with DxDiag after Windows restart'} | ConvertTo-Json -Depth 5 -Compress

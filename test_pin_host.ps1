#Requires -Version 5.1
<#
.SYNOPSIS
    在 CUDA 容器中测试 cudaMallocHost 锁定内存(pinned memory)的可分配上限。

.EXAMPLE
    .\test_pin_host.ps1                       # 默认 2 GiB（预期失败）
.EXAMPLE
    .\test_pin_host.ps1 -SizeGb 1             # 1 GiB（预期成功）
.EXAMPLE
    .\test_pin_host.ps1 -SizeGb 0.5 -Repeat 4 # 4 x 0.5 GiB，同时持有
.EXAMPLE
    .\test_pin_host.ps1 -Probe -SizeGb 4      # 二分查找单次分配上限
.EXAMPLE
    .\test_pin_host.ps1 -Budget               # 探测 pinned 总量上限
.EXAMPLE
    .\test_pin_host.ps1 -SizeGb 1 -Cycles 4   # 反复申请/释放 1 GiB
.EXAMPLE
    .\test_pin_host.ps1 -SizeGb 2 -Mode register   # 改用 cudaHostRegister
.EXAMPLE
    .\test_pin_host.ps1 -SizeGb 2 -Image "nvcr.io/nvidia/pytorch:24.07-py3"
#>
param(
    [double]$SizeGb = 2,
    [int]$Repeat = 1,
    [switch]$Probe,
    [switch]$Budget,
    [ValidateSet("malloc", "register")]
    [string]$Mode = "malloc",
    [int]$Cycles = 1,
    [string]$Image = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    [string]$ShmSize = "8g",
    [string]$Memlock = "-1:-1"
)

$ErrorActionPreference = "Stop"

if ($SizeGb -le 0) { throw "-SizeGb must be > 0" }
if ($Repeat -lt 1) { throw "-Repeat must be >= 1" }
if ($Cycles -lt 1) { throw "-Cycles must be >= 1" }

# 使用不变文化格式化，避免小数点变成逗号（如 0,5）导致容器解析失败
$sizeArg = $SizeGb.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$scriptPath = Join-Path $PSScriptRoot "test_pin_host.py"

Write-Host "===== size_gb=$sizeArg repeat=$Repeat mode=$Mode probe=$($Probe.IsPresent) budget=$($Budget.IsPresent) image=$Image =====" -ForegroundColor Cyan

$dockerArgs = @(
    "run", "--rm", "--gpus", "all",
    "--ipc=host",
    "--shm-size=$ShmSize",
    "--ulimit", "memlock=$Memlock",
    "-e", "PIN_SIZE_GB=$sizeArg",
    "-e", "PIN_REPEAT=$Repeat",
    "-e", "PIN_MODE=$Mode",
    "-e", "PIN_CYCLES=$Cycles",
    "-v", "${scriptPath}:/var/local/test_pin_host.py",
    "-w", "/var/local",
    $Image,
    "python3", "/var/local/test_pin_host.py",
    "--size-gb", $sizeArg,
    "--repeat", "$Repeat",
    "--mode", $Mode,
    "--cycles", "$Cycles"
)

if ($Probe) { $dockerArgs += "--probe" }
if ($Budget) { $dockerArgs += "--budget" }

& docker @dockerArgs
exit $LASTEXITCODE
  
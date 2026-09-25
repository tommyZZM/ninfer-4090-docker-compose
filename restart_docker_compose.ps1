# 交互式启动/重启 NInfer-4090 服务 (Windows PowerShell)
# 用法:
#   .\restart_docker_compose.ps1                                  # 列出 models 目录中的模型, 选择后启动
#   .\restart_docker_compose.ps1 -Build                           # 启动前先重新构建镜像
#   .\restart_docker_compose.ps1 -Recreate                         # 强制重建容器 (即使配置未变化)
#   .\restart_docker_compose.ps1 -DryRun                          # 只打印将执行的命令, 不实际运行
#   .\restart_docker_compose.ps1 -ModelFile models\xxx.ninfer     # 跳过交互, 直接指定

param(
    [switch]$Build,
    [switch]$Recreate,
    [switch]$DryRun,
    [string]$ModelFile
)

$ErrorActionPreference = "Stop"
$modelsDir = Join-Path $PSScriptRoot "models"
$composeArgs = @("compose", "up", "-d")
if ($Build) { $composeArgs += "--build" }
if ($Recreate) { $composeArgs += "--force-recreate" }

if (-not (Test-Path $modelsDir)) {
    Write-Error "未找到 models 目录: $modelsDir"
    exit 1
}

# ---------- 确定要加载的模型 ----------
if ([string]::IsNullOrWhiteSpace($ModelFile)) {
    $models = @(Get-ChildItem -Path $modelsDir -Filter "*.ninfer" -File | Sort-Object Name)
    if ($models.Count -eq 0) {
        Write-Error "models 目录中没有 .ninfer 模型文件: $modelsDir"
        exit 1
    }

    Write-Host "可用模型 (models 目录):" -ForegroundColor Cyan
    for ($i = 0; $i -lt $models.Count; $i++) {
        $sizeMB = [math]::Round($models[$i].Length / 1MB, 1)
        Write-Host ("  [{0}] {1}  ({2} MB)" -f ($i + 1), $models[$i].Name, $sizeMB)
    }
    Write-Host "  [0] 手动输入模型文件名"

    $choice = (Read-Host "请选择模型编号").Trim()
    if ($choice -eq "0") {
        $ModelFile = (Read-Host "模型文件名 (如 qwen3_8_27b_3526913004b1.ninfer)").Trim()
    }
    elseif ($choice -match '^\d+$' -and [int]$choice -ge 1 -and [int]$choice -le $models.Count) {
        $ModelFile = $models[[int]$choice - 1].Name
    }
    else {
        Write-Error "无效选择: $choice"
        exit 1
    }
}

# 校验文件存在 (允许带或不带 models/ 前缀)
$candidate = $ModelFile -replace '^\\?/?models[/\\]', ''
if (-not (Test-Path (Join-Path $modelsDir $candidate))) {
    Write-Error "模型文件不存在: $modelsDir\$candidate"
    exit 1
}

# 容器内路径: 相对 /workspace, 即 models/<文件名>
$env:MODEL_FILE = "models/$candidate"
Write-Host "已选择模型: $($env:MODEL_FILE)" -ForegroundColor Green

# ---------- 启动 docker compose ----------
Push-Location $PSScriptRoot
try {
    Write-Host "执行: docker $($composeArgs -join ' ')" -ForegroundColor Cyan
    if ($DryRun) {
        Write-Host "[dryrun] 未实际执行" -ForegroundColor Yellow
    } else {
        & docker @composeArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "docker compose 失败 (exit code: $LASTEXITCODE)"
            exit $LASTEXITCODE
        }
        Write-Host "服务已启动: http://127.0.0.1:11439/v1" -ForegroundColor Green
        Write-Host "查看日志: docker compose logs -f"
        Write-Host "停止服务: docker compose down"
    }
}
finally {
    Pop-Location
}

# 构建 ninfer-4090:sm89 镜像 (CUDA 13.1, sm_89 / RTX 4090)
# 用法: .\build_ninfer_4090.ps1
# 说明: 从 ninfer-4090 子目录读取 Dockerfile, 构建后打 tag ninfer-4090:sm89

$srcDir = Join-Path $PSScriptRoot "ninfer-4090"

if (-not (Test-Path (Join-Path $srcDir "Dockerfile"))) {
    Write-Error "未找到 Dockerfile: $srcDir"
    exit 1
}

Push-Location $srcDir
try {
    docker build --tag ninfer-4090:sm89 .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "docker build 失败 (exit code: $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    Write-Host "构建成功: ninfer-4090:sm89" -ForegroundColor Green
}
finally {
    Pop-Location
}

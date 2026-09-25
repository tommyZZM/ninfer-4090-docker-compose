#!/usr/bin/env bash
# 交互式启动/重启 NInfer-4090 服务 (Linux / macOS / WSL)
# 用法:
#   ./restart_docker_compose.sh                                  # 列出 models 目录中的模型, 选择后启动
#   ./restart_docker_compose.sh --build                          # 启动前先重新构建镜像
#   ./restart_docker_compose.sh --recreate                       # 强制重建容器 (即使配置未变化)
#   ./restart_docker_compose.sh --dryrun                         # 只打印将执行的命令, 不实际运行
#   ./restart_docker_compose.sh --model-file models/xxx.ninfer   # 跳过交互, 直接指定
#   MODEL_FILE=models/xxx.ninfer docker compose up -d            # 不用脚本, 直接传环境变量
# 兼容 sh 调用: 若被 dash/sh 执行, 自动切换到 bash 重新运行本脚本

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$ROOT/models"
DRYRUN=0
MODEL_FILE=""
ARGS=(compose up -d)

# ---------- 解析参数 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)      ARGS+=(--build); shift ;;
        --recreate)   ARGS+=(--force-recreate); shift ;;
        --dryrun)     DRYRUN=1; shift ;;
        --model-file)
            [[ $# -ge 2 ]] || { echo "错误: --model-file 需要一个参数" >&2; exit 1; }
            MODEL_FILE="$2"; shift 2 ;;
        -h|--help)    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            echo "未知参数: $1 (使用 --help 查看用法)" >&2; exit 1 ;;
    esac
done

[[ -d "$MODELS_DIR" ]] || { echo "错误: 未找到 models 目录: $MODELS_DIR" >&2; exit 1; }

# ---------- 确定要加载的模型 ----------
if [[ -z "$MODEL_FILE" ]]; then
    # 收集 models 目录下的 .ninfer 文件 (glob 默认按名称排序)
    MODELS=("$MODELS_DIR"/*.ninfer)
    [[ -e "${MODELS[0]}" ]] || { echo "错误: models 目录中没有 .ninfer 模型文件: $MODELS_DIR" >&2; exit 1; }

    echo "可用模型 (models 目录):"
    for i in "${!MODELS[@]}"; do
        printf "  [%d] %s  (%s MB)\n" "$((i + 1))" "$(basename "${MODELS[$i]}")" "$(du -m "${MODELS[$i]}" | cut -f1)"
    done
    echo "  [0] 手动输入模型文件名"

    read -rp "请选择模型编号: " choice
    if [[ "$choice" == "0" ]]; then
        read -rp "模型文件名 (如 qwen3_8_27b_3526913004b1.ninfer): " MODEL_FILE
    elif [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#MODELS[@]} )); then
        MODEL_FILE="$(basename "${MODELS[$((choice - 1))]}")"
    else
        echo "错误: 无效选择: $choice" >&2
        exit 1
    fi
fi

# 归一化: 去掉可能带上的 models/ 或 ./ 前缀, 并校验存在
MODEL_FILE="${MODEL_FILE#models/}"; MODEL_FILE="${MODEL_FILE#./}"
[[ -f "$MODELS_DIR/$MODEL_FILE" ]] || { echo "错误: 模型文件不存在: $MODELS_DIR/$MODEL_FILE" >&2; exit 1; }

# 容器内路径: 相对 /workspace, 即 models/<文件名>
export MODEL_FILE="models/$MODEL_FILE"
echo "已选择模型: $MODEL_FILE"

# ---------- 启动 docker compose ----------
cd "$ROOT"
echo "执行: docker ${ARGS[*]}"
if [[ "$DRYRUN" -eq 1 ]]; then
    echo "[dryrun] 未实际执行"
else
    docker "${ARGS[@]}"
    echo "服务已启动: http://127.0.0.1:11439/v1"
    echo "查看日志: docker compose logs -f"
    echo "停止服务: docker compose down"
fi
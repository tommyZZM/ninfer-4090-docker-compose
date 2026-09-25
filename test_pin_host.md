# WSL2 + Docker 下 CUDA pinned memory（cudaMallocHost）上限调查报告

调试脚本：[`test_pin_host.ps1`](./test_pin_host.ps1)（docker 包装）+ [`test_pin_host.py`](./test_pin_host.py)（探测逻辑）

---

## 1. 结论（TL;DR）

1. **问题不在 `size_gb = 1` / `2` 这条线上，也不在 `cudaMallocHost` 的用法上。**
2. WSL2 的 GPU 驱动只给**整个 VM 一个全局的 pinned memory 预算**，实测为 **≈ 1.11~1.125 GiB**（区间 `[1136, 1152) MiB`）。超出即返回 `cudaErrorMemoryAllocation`（错误码 2，`out of memory`）。
3. 该预算是 **全局共享**的：与进程、容器、以及**具体 pin 的 API（`cudaMallocHost` / `cudaHostRegister`）都无关**。
4. 因此 **在不改用其他机制的前提下，2 GiB 的 pinned 分配无法成功**，也不是 `--ulimit memlock`、`.wslconfig memory=`、`--ipc=host`、`--shm-size` 能解决的。
5. 可行方案只有三类：**≤ 1 GiB 的环形/窗口复用**（已验证）、**改用页内存 + `cudaMemcpyAsync` 由驱动内部 staging**、或**在原生 Linux 上运行**。

---

## 2. 复现环境

| 项 | 值 |
| --- | --- |
| 宿主机 | Windows，物理内存 **63.92 GiB** |
| WSL2 VM | `MemTotal = 15.62 GiB`（`.wslconfig`: `memory=16GB`, `swap=8GB`） |
| WSL 内核 | `6.6.87.2-microsoft-standard-WSL2` |
| GPU | `NVIDIA GeForce RTX 4090 D`, 24564 MiB VRAM |
| 驱动 | `NVIDIA-SMI 615.78.02`，`KMD Version: 617.14`，`CUDA UMD Version: 13.4` |
| 容器镜像 | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`（Python 3.11.9，`libcudart.so.12`） |
| 启动参数 | `--gpus all --ipc=host --shm-size=8g --ulimit memlock=-1:-1` |

原始现象：

```text
WSL2 Memory (Total):    15.62 GiB
WSL2 Memory (Available): 14.62 GiB
=== (RLIMIT_MEMLOCK) ===
Soft Limit: Unlimited
Hard Limit: Unlimited
FAILED to allocate 2048 MiB: out of memory
```

→ 内存明明富余、memlock 也无限，2 GiB 仍然失败。

---

## 3. 实测数据

### 3.1 单次分配上限（`--probe`）

`.\test_pin_host.ps1 -Probe -SizeGb 4`

| 请求 | 结果 |
| --- | --- |
| 2.000 GiB | fail |
| 1.000 GiB | ok |
| 1.500 GiB | fail |
| 1.250 GiB | fail |
| 1.125 GiB | fail |
| 1.0625 GiB | **ok** |

→ 单次分配上限落在 **1.0625 ~ 1.125 GiB** 之间。

### 3.2 总量预算（关键实验）

| 实验 | 成功量 | 结果 |
| --- | --- | --- |
| 0.5 GiB × 2 | 1.00 GiB | 成功 |
| 0.5 GiB × 3 | 1.00 GiB | **第 3 块失败** |
| 64 MiB × 17 | 1088 MiB | 第 18 块（累计 1152 MiB）失败 |
| 1 MiB × 1120 | 1120 MiB | **第 1121 块失败** |
| 16 MiB × 71 | **1136 MiB** | **第 72 块（累计 1152 MiB）失败** |
| `-Mode register`，16 MiB × 71 | **1136 MiB** | 与 `malloc` 完全一致 |

**结论：总预算 ≈ 1136–1152 MiB（约 1.11~1.125 GiB），与块大小无关。**
（1 MiB 分块只到 1120 MiB，说明每次分配还有约 ~14 KiB 的元数据开销。）

### 3.3 预算是全局共享的（跨进程 / 跨容器）

| 实验 | 结果 |
| --- | --- |
| 两个容器**并发**各申请 1 GiB | **只有一个成功** |
| 两个容器**并发**各申请 0.5 GiB | 两个都成功 |
| 1 GiB 连续 申请→释放 4 轮（`-Cycles 4`） | 全部成功（预算可回收复用） |

→ 预算不是 per-process / per-container，而是 **per WSL2 VM（设备级全局池）**；
同时它**可回收**，所以「复用 ≤ 1 GiB 缓冲」是可行的。

### 3.4 与 pin 方式无关

| API | 2 GiB 单次 | 预算 |
| --- | --- | --- |
| `cudaMallocHost`（`-Mode malloc`） | fail | `[1136, 1152)` MiB |
| `cudaHostRegister`（`-Mode register`） | fail | `[1136, 1152)` MiB |

→ 换 API **不能**绕过该限制，说明限制作用在 WSL2 的「页锁定」这个能力层面。

---

## 4. 已排除的原因

| 假设 | 实测证据 | 结论 |
| --- | --- | --- |
| `RLIMIT_MEMLOCK` 太小 | Soft/Hard 均为 **Unlimited**；`--ulimit memlock=-1:-1` 无变化 | ❌ 排除 |
| 物理内存不足 | `MemAvailable ≈ 14.6 GiB`，而只申请 2 GiB | ❌ 排除 |
| 内核内存 overcommit 限制 | `vm.overcommit_memory = 1`（always overcommit） | ❌ 排除 |
| mlock 记账限制 | `/proc/meminfo: Mlocked = 0 kB` | ❌ 排除 |
| cgroup 内存限制 | `cgroup v2 memory.max = max`（v1 不存在） | ❌ 排除 |
| 共享内存 / IPC 配置 | `--ipc=host` 与否都是 1088 MiB；`--shm-size=8g` 与 `64m` 结果相同 | ❌ 排除 |
| WSL2 VM 内存配小了 | 用户已确认改 `.wslconfig` 的 `memory=` **无论怎么变都没效果** | ❌ 排除 |
| 代码/API 用法问题 | `cudaMallocHost` 与 `cudaHostRegister` 撞同一个池 | ❌ 排除 |

---

## 5. 根因

**这是 WSL2 的已知限制，不是配置错误。**

- 在 WSL2（GPU-PV）下，pinned host memory 需要由 Windows 侧内核态驱动（KMD 617.14）经 `dxgkrnl` 完成分配与锁定，驱动为此维护一个**固定大小的全局池**（本机实测 ≈ 1.12 GiB），**不随 VM 内存、空闲内存、memlock、cgroup 变化**。
- NVIDIA 官方《CUDA on WSL User Guide》第 5.1 节 *Known Limitations for Linux CUDA Applications* 明确列出：

  > **Pinned system memory** (example: System memory that an application makes resident for GPU accesses) **availability for applications is limited.** For example, some deep learning training workloads, depending on the framework, model and dataset size used, can exceed this limit and may not work.

- 因此「1 GB 能成功、2 GB 失败」的真实分界并不是 1 GiB，而是 **≈ 1.12 GiB 的全局池**。

---

## 6. 修复 / 规避方案

按推荐程度排序：

### ✅ 方案 A：≤ 1 GiB 的 pinned 环形缓冲（已验证）

把大缓冲拆成一个或多个 pinned 窗口（建议单块 ≤ 1 GiB，留出余量给同 VM 内其他使用者），**循环复用**而不是同时持有 2 GiB。

```powershell
.\test_pin_host.ps1 -SizeGb 1 -Cycles 4   # 1 GiB 缓冲申请/释放 4 轮，全部成功
```

适用：分块传输、KV/权重分片搬运、turn-checkpoint 之类的流式场景。

### ✅ 方案 B：页内存 + `cudaMemcpyAsync`

保持 2 GiB 为普通 malloc 内存，用 `cudaMemcpyAsync`（驱动内部自行 staging），不显式 pin。
代价：带宽低于真正的 pinned，但**不受 1.12 GiB 限制**，任意大小可用。

### ✅ 方案 C：在原生 Linux 上运行

裸机 Linux **没有**这个 pinned 池限制（`--ulimit memlock` 才真正生效）。如果必须长期持有 >1 GiB pinned，这是唯一稳定解。

### ⚠️ 方案 D：升级 Windows NVIDIA 驱动

属 NVIDIA 文档化的 WSL2 限制，升级驱动**不保证**放宽；可尝试但不作为依赖。

### ❌ 无效的做法（不要浪费时间）

- 调大 `.wslconfig` 的 `memory=` / `swap=`（用户已实测无效）
- `--ulimit memlock=-1:-1`（本机已是 Unlimited，仍然只在 ~1.12 GiB）
- 改用 `cudaHostRegister`（撞同一个池）
- 把 2 GiB 拆成多个小块同时持有（**总量**受限，拆块无效）
- 调整 `--ipc=host` / `--shm-size`
- 调大 Docker Desktop 的 `MemoryMiB`

---

## 7. 对 ninfer-4090 部署的注意事项

1. 该池是**全局**的：若同时跑两个各要 1 GiB pinned 的容器，**必有一个失败**。多容器/多进程部署时必须显式分配预算。
2. 建议把应用内 pinned 缓冲总量控制在 **≤ 1 GiB**，其余走方案 B。
3. 该限制与 `--gpus`、镜像版本无关（不同镜像上测得的池大小可能不同，可用 `-Budget` 复测）。

---

## 8. 脚本用法

两个文件均已改为**动态参数**（不再需要手改 `size_gb = 2`）。

### `test_pin_host.ps1`

```powershell
.\test_pin_host.ps1                              # 默认 2 GiB（会失败，用于演示）
.\test_pin_host.ps1 -SizeGb 1                    # 单个 1 GiB
.\test_pin_host.ps1 -SizeGb 0.5 -Repeat 4        # 同时持有 4 × 0.5 GiB
.\test_pin_host.ps1 -Probe -SizeGb 4             # 二分查找单次分配上限
.\test_pin_host.ps1 -Budget                      # 逐块探测 pinned 总量上限
.\test_pin_host.ps1 -SizeGb 1 -Cycles 4          # 验证缓冲可复用
.\test_pin_host.ps1 -SizeGb 2 -Mode register     # 改用 cudaHostRegister 对比
.\test_pin_host.ps1 -SizeGb 2 -Image "nvcr.io/nvidia/pytorch:24.07-py3"
```

| 参数 | 说明 |
| --- | --- |
| `-SizeGb <double>` | 单个 pinned 块大小（GiB），默认 `2` |
| `-Repeat <int>` | 同时持有多少块，默认 `1` |
| `-Cycles <int>` | 申请/释放轮数，`>1` 用于验证复用，默认 `1` |
| `-Mode malloc\|register` | `cudaMallocHost` 或 `cudaHostRegister`，默认 `malloc` |
| `-Probe` | 二分查找单次分配上限 |
| `-Budget` | 用 16 MiB 分块探测总量上限 |
| `-Image` / `-ShmSize` / `-Memlock` | 覆盖镜像与 docker 参数 |

退出码：成功 `0`，失败 `1`（脚本会 `exit $LASTEXITCODE`）。

### `test_pin_host.py`（容器内直接用）

```bash
python3 test_pin_host.py --size-gb 2 --repeat 1
python3 test_pin_host.py --budget
python3 test_pin_host.py --probe --size-gb 4 --mode register
```

环境变量（便于 `docker run -e`）：`PIN_SIZE_GB`、`PIN_REPEAT`、`PIN_MODE`、`PIN_CYCLES`、`LIBCUDART_PATH`。

---

## 9. 复现步骤（一条命令）

```powershell
.\test_pin_host.ps1 -Budget                      # 1) 量出全局上限 ≈ 1136–1152 MiB
.\test_pin_host.ps1 -SizeGb 2                    # 2) 2 GiB 单次失败（cap 之上）
.\test_pin_host.ps1 -SizeGb 2 -Mode register     # 3) 换 API 仍失败 → 与 API 无关
.\test_pin_host.ps1 -SizeGb 1 -Cycles 4          # 4) 复用 1 GiB 可行 → 方案 A
```

---

## 10. 参考资料

- NVIDIA, *CUDA on WSL User Guide*, §5.1 Known Limitations for Linux CUDA Applications
  （"Pinned system memory ... availability for applications is limited."）
  <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>

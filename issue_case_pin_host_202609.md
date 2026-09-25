```
docker run --rm --gpus all --publish 8080:8080   --volume "$PWD/models:/workspace/models:ro"   ninfer-4090:sm89   ninfer-serve models/qwen3_8_27b_3526913004b1.ninfer   --ho
st 0.0.0.0 --port 8080   --max-context 262144 --kv-capacity 262144   --max-concurrency 1 --max-pending-requests 16   --p
ending-timeout-ms 600000   --prefill-chunk 1024 --kv-dtype rk4v4-e8   --spec mtp --draft-tokens 3 --lm-head-draft   --vi
sion --preserve-thinking

==========
== CUDA ==
==========

CUDA Version 13.1.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

2026-09-25 00:11:42.855  INFO  starting engine
2026-09-25 00:11:43.275  INFO  loading weights | 16.9 GiB
2026-09-25 00:11:53.325  INFO    loading weights 8.9% | 1.50 GiB/16.9 GiB | 168.2 MiB/s | ETA 1m 34.0s
2026-09-25 00:12:03.613  INFO    loading weights 18.4% | 3.12 GiB/16.9 GiB | 163.3 MiB/s | ETA 1m 26.7s
2026-09-25 00:12:13.704  INFO    loading weights 28.0% | 4.75 GiB/16.9 GiB | 166.0 MiB/s | ETA 1m 15.2s
2026-09-25 00:12:24.061  INFO    loading weights 38.0% | 6.44 GiB/16.9 GiB | 168.3 MiB/s | ETA 1m 3.9s
2026-09-25 00:12:34.138  INFO    loading weights 47.6% | 8.06 GiB/16.9 GiB | 165.0 MiB/s | ETA 55.2s
2026-09-25 00:12:44.258  INFO    loading weights 57.2% | 9.69 GiB/16.9 GiB | 158.2 MiB/s | ETA 47.0s
2026-09-25 00:12:54.280  INFO    loading weights 66.7% | 11.3 GiB/16.9 GiB | 167.2 MiB/s | ETA 34.5s
2026-09-25 00:13:04.357  INFO    loading weights 76.3% | 12.9 GiB/16.9 GiB | 163.6 MiB/s | ETA 25.1s
2026-09-25 00:13:14.720  INFO    loading weights 86.3% | 14.6 GiB/16.9 GiB | 168.5 MiB/s | ETA 14.1s
2026-09-25 00:13:24.732  INFO    loading weights 95.9% | 16.2 GiB/16.9 GiB | 169.2 MiB/s | ETA 4.2s
2026-09-25 00:13:29.211  INFO  weights ready | 16.9 GiB | 1m 45.9s | 163.8 MiB/s
2026-09-25 00:13:29.962  INFO  pinning host state | 1.15 GiB
2026-09-25 00:13:34.482  ERROR startup failed | pinning host state | 4.5s
2026-09-25 00:13:34.726  FATAL server failed during startup | cudaMallocHost failed: cudaErrorMemoryAllocation: out of memory
```

---

# 补充（2026-09-25）：从 ninfer-4090 源码定位到的根因与修复

## A. 根因定位（代码级证据）

`pinning host state | 1.15 GiB` 是 `HostStatePool` 的 pinned 后端分配：

| 环节 | 代码位置 | 说明 |
| --- | --- | --- |
| pinned 原语 | [`src/core/arena.cu:244`](./ninfer-4090/src/core/arena.cu#L244) | `PinnedHostBuffer` → `cudaMallocHost`；失败抛 `cudaMallocHost failed: ... out of memory`，与日志 FATAL 一致 |
| 默认容量 | [`include/ninfer/types.h:153-154`](./ninfer-4090/include/ninfer/types.h#L153-L154) | `host_state_slots = 8`，`host_kv_capacity_bytes = 8 GiB` |
| Host State 分配 | [`src/targets/qwen3_6/impl/runtime/program_impl.h:851-859`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851-L859) | `host_state_bytes = image_bytes × host_state_slots` → `HostStatePool` |
| Host State 后端 | [`src/targets/qwen3_6/impl/state/state_image.cpp:212-225`](./ninfer-4090/src/targets/qwen3_6/impl/state/state_image.cpp#L212-L225) | `backing_.emplace(bytes)` = `cudaMallocHost` |
| 单槽大小 | [`docs/turn-checkpoint-ring.md`](./ninfer-4090/docs/turn-checkpoint-ring.md) | Qwen3.8-27B 一份 StateImage ≈ 147 MiB |
| Host KV 分配 | [`src/targets/qwen3_6/impl/runtime/program_impl.h:911-928`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911-L928) | `HostKVArena(capacity)` → [`src/core/host_kv_arena.cpp:231`](./ninfer-4090/src/core/host_kv_arena.cpp#L231) 的 `backing_.emplace` = 又一次 `cudaMallocHost` |
| 解析入口 | [`src/serve/serve_options.cpp:252`](./ninfer-4090/src/serve/serve_options.cpp#L252)、[`:260`](./ninfer-4090/src/serve/serve_options.cpp#L260) | `--host-state-slots` / `--host-kv-mib` |
| 文档默认值 | [`docs/serving.md:840-841`](./ninfer-4090/docs/serving.md#L840-L841) | `--host-state-slots` 默认 `8`；`--host-kv-mib` 默认 `8192` |

数值核对：

```text
host state pinned = 8 × 147.2 MiB = 1177.6 MiB = 1.148 GiB  （正好等于日志 1.15 GiB）
WSL2 全局 pinned 池实测上限 = [1136, 1152) MiB
=> 1.148 GiB 恰好越界，必然 cudaMallocHost OOM
```

结论：这是**默认 pinned 容量 > WSL2 全局池上限**，不是 `cudaMallocHost` 用法问题，
也不是 `.wslconfig` / `memlock` / `shm` / `ipc` 配置问题（与本目录 [`test_pin_host.md`](./test_pin_host.md) 的实测结论一致）。
`ninfer-serve` 在该阶段没有降级/回退逻辑，申请失败即 `throw` → FATAL 退出。

### 附带风险：Host KV 默认 8 GiB 是"下一颗地雷"

即使把 host state 压到上限以下，紧随其后的 `HostKvPin` 阶段仍会按默认
`host_kv_capacity_bytes = 8 GiB` 再申请一次 pinned，同样必然失败。
**`--host-state-slots` 与 `--host-kv-mib` 是相互独立的两块 pinned 容量，必须同时下调。**

（另外：[`src/artifact/materializer.cpp`](./ninfer-4090/src/artifact/materializer.cpp#L183-L266) 的权重 staging 用 64 MiB × ≤4 pinned，但在
`pinning host state` 之前已 `slots.clear()` 释放，不占用本次预算；`PagedKVCache::host_shadow_`
（[`src/core/paged_kv_cache.h:377`](./ninfer-4090/src/core/paged_kv_cache.h#L377)）只有 MB 量级。）

### 与 4090D 的关系

pinned host memory 位于 Windows 侧驱动（KMD 617.14 / dxgkrnl）与 WSL2 VM 的宿主内存路径，
与 GPU 型号（4090 / 4090D / 5090）无关。因此 4090D 与上游 4090 的差异**不会**改变本
问题的边界；上限只由 WSL2 驱动的全局池大小决定。

### `--host-state-slots` 与 `--host-kv-mib` 分别是做什么的

这两个参数都只服务于 **context cache 的降级（demote）能力**——把 inactive 的 checkpoint
从显存挪到 pinned 主机内存，从而在显存吃紧时"保命"而不是直接丢弃。它们**不扩大**并发数、
不扩大单请求上下文上限，也不参与 active 请求的执行路径。

| 参数 | 承载什么 | 计量单位 | 默认值 | 对应源码 |
| --- | --- | --- | --- | --- |
| `--host-state-slots` | **完整 StateImage 的副本** = 混合模型的 GDN/线性注意力 recurrent state + conv state + boundary hidden。这是"可继续演化的会话状态"，每份约 **147 MiB**（Qwen3.8-27B） | 槽位个数（× 每份 image_bytes） | `8` | [`include/ninfer/types.h:153`](./ninfer-4090/include/ninfer/types.h#L153) → [`state_image.cpp:212-225`](./ninfer-4090/src/targets/qwen3_6/impl/state/state_image.cpp#L212-L225) |
| `--host-kv-mib` | **typed packed KV 页的 Host 副本**，由 Main Text KV 与 selected backend（MTP/DFlash）pool **共用**同一个 arena，按物理 page extent 切分 | MiB（字节容量） | `8192`（8 GiB） | [`include/ninfer/types.h:154`](./ninfer-4090/include/ninfer/types.h#L154) → [`host_kv_arena.cpp:231`](./ninfer-4090/src/core/host_kv_arena.cpp#L231) |

#### 为什么是"两块独立容量"

State 与 KV 是**独立 placement** 的资源（`docs/maintainer/resource-scheduling-and-context-cache.md`
§5.1/§5.2）：一个 checkpoint 可能 State 在 Host、KV 在 Device，也可能反过来。因此
`Host StateImages` 与 `Host KV bytes` 必须分别计费，二者**不可互相替代**：

- StateImage 是**完整迁移单位**，不做部分 demote；一份就要占满一个 slot。
- Host KV 是**按 extent 分配**的字节池，Main/Backend 共享，容量按 page_stride × pages 计。

#### 它们实际被谁使用

| 使用点 | 源码位置 |
| --- | --- |
| 启动时按槽数分配 pinned StateImage 池 | [`program_impl.h:851-859`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851-L859) |
| 启动时按字节分配 pinned Host KV arena | [`program_impl.h:911-928`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911-L928) |
| 压力规划中把 Device-only 的 State **降级到 Host**（`DemoteSharedToHost` / `DemoteEndpointToHost` / `DemoteRewriteToHost`） | [`program_impl.h:1468-1479`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L1468-L1479)、[`1663-1704`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L1663-L1704) |
| 压力规划中把 KV page **降级到 Host**（`PressureKVDecisionKind::DemoteToHost`） | [`program_impl.h:1492-1501`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L1492-L1501)、[`2669-2681`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L2669-L2681) |
| 三档 placement 语义：`DeviceOnly` / `HostOnly` / `Both` | [`program_impl.h:2214-2220`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L2214-L2220) |
| Host KV extent 的所有权与页成员关系 | [`host_kv_extent_store.h`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/host_kv_extent_store.h) |
| 架构定义（"Host State 与 Host KV 独立计费"） | [`docs/maintainer/resource-scheduling-and-context-cache.md`](./ninfer-4090/docs/maintainer/resource-scheduling-and-context-cache.md) §11 |
| 面向用户的说明（"Host KV is shared by Main and the selected Backend pool and is consumed in physical page extents"） | [`docs/serving.md:871-874`](./ninfer-4090/docs/serving.md#L871-L874) |

#### 与本次故障的关系

正因为它们是**独立的 pinned 容量**、且默认值分别为 `8 × 147 MiB` 与 `8 GiB`，才会出现：

```text
Host State  8 × 147.2 MiB = 1.148 GiB   ← 先撞上限，日志停在 "pinning host state"
Host KV     8 GiB         = 8192 MiB    ← 即使前者压下去，这里仍会撞
```

**两者必须一起下调**才能启动。把它们设为 `0` 的语义是：放弃 Host 降级能力，checkpoint
只能留在 Device 或直接被 evict/丢弃——**active 请求的执行与正确性不受影响**，只是失去了
显存压力下的"缓存保命"余地（见 `docs/maintainer/...md` §11 最后一段：context cache
disabled 时采用 root-only 语义，Device/Host checkpoint 容量为零）。

## B. 修复方案（按推荐度排序）

记 \(B \approx 1100\) MiB 为可用 pinned 预算（给同 VM 其他使用者留 ~50 MiB 余量），需满足：

\[
\text{host\_state\_slots} \times 147\ \text{MiB} + \text{--host-kv-mib} \le B
\]

### 方案 1（最稳、零 pinned）：`--no-prefix-reuse`

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b_3526913004b1.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --preserve-thinking \
  --no-prefix-reuse
```

- 效果：[`src/serve/serve_options.cpp:385-388`](./ninfer-4090/src/serve/serve_options.cpp#L385-L388) 把 `context_cache.enabled = false`、`host_state_slots = 0`、`host_kv_capacity_bytes = 0`，
  两个 pin 阶段整体跳过。
- 代价：关闭 context cache（无跨请求 prefix 复用、无 continuation 保留）。
- 注意：`--no-prefix-reuse` **不能与任何显式 context-cache 容量参数共存**（含 0 值），见 [`docs/serving.md`](./ninfer-4090/docs/serving.md)。

### 方案 2（推荐折中）：保留 context cache，但走纯 Device、不占 pinned

```bash
  ... --host-state-slots 0 --host-kv-mib 0
```

- 效果：context cache 仍启用、device checkpoint 仍可用；只是 inactive continuation
  无法 demote 到 Host。两个 pin 阶段因为容量为 0 而跳过。
- 适合单并发多轮对话：prefix 复用仍生效，只是没有 pinned host 兜底。

### 方案 3（需要部分 pinned 兜底）：把总量压到预算内

```bash
  # ---- 3a：保留 1 个 Host State 槽 + 768 MiB Host KV（合计 ≈ 915 MiB）----
  #
  # 【智力 / 输出质量】无任何影响。
  #   这两个参数只决定 pinned host 缓存的大小，不碰模型权重、数值精度、KV dtype
  #   （--kv-dtype）、采样参数，也不改变 --max-context / --kv-capacity。
  #   缓存命中与否只决定"要不要重算"，不决定"算出来是什么"：checkpoint 恢复是
  #   精确快照回放（exact identity，见架构文档 §4.4/§4.5），因此同一 prompt 的
  #   token 级输出与 8 槽默认配置完全一致，不会"变笨"。
  #
  # 【对话过程】影响的是多轮对话的 TTFT（首 token 延迟），不是回答内容。
  #   - 为什么优先保 State：Qwen3.8-27B 里 48 层是 GDN 线性注意力，其 recurrent
  #     state 逐 token 积分、**无法回退**，只能从"曾经复制下来的快照"继续；
  #     而 16 层 full-attention 的 KV 是可定位的，可以截断后重算。
  #     所以 StateImage 才是不可再生资源，KV 是可重算的。
  #   - 1 个 Host State 槽 = 最多让 1 条 inactive continuation 的 State 离开显存。
  #   - 768 MiB Host KV 用于把这些 checkpoint 的 KV 页也搬离显存；Main 与 selected
  #     backend（MTP/DFlash）共用这一个池，按物理 page extent 计费。
  #   - 三者（State 副本 + 每页 KV 副本）有一个不满足，该 checkpoint 就失效。
  #   - 结果：常规多轮对话命中率基本够用；但"改写历史中部"这类需要较深 anchor
  #     的场景，可能因 anchor 已被 evict 而退回 root 全量 re-prefill（TTFT 明显变长）。
  ... --host-state-slots 1 --host-kv-mib 768     # ≈ 147 + 768 = 915 MiB

  # ---- 3b：不留 Host State 槽，全部预算给 Host KV（= 1024 MiB）----
  #
  # 【智力 / 输出质量】同样无影响（理由同 3a：只影响缓存，不影响计算内容与精度）。
  #
  # 【对话过程】与 3a 是两种取舍，不是"谁更好"：
  #   - host_state_slots = 0 → StateImage 无处可去，必须常驻 Device；显存吃紧时
  #     只能整条 checkpoint 被 evict/丢弃，**无法降级保命**。
  #   - 好处是 Host KV 更宽裕（1024 MiB），可以把更多 KV 页搬离显存，反而给
  #     Device 腾出空间放 StateImage —— 对"KV 特别长、State 本来就常驻"的
  #     单并发长上下文场景可能更有利。
  #   - 代价是失去了"保 State"的能力：一旦 Device State 被挤掉，就需要从 root
  #     重新 prefill 才能重建那段不可回退的 recurrent state。
  #   - 经验判断：面对"多轮对话 + 历史改写"（本 issue 的使用场景），3a 通常更稳，
  #     因为它保住的是不可重算的 State；3b 更适合纯长上下文、少改写的场景。
  ... --host-state-slots 0 --host-kv-mib 1024    # = 1024 MiB
```

- 校验：`1 × 147.2 + 768 = 915 MiB ≤ 1100 MiB`，安全；`0 × 147.2 + 1024 = 1024 MiB ≤ 1100 MiB`，安全。
- 注意：WSL2 池是 **VM 全局**的，若同机还跑别的要求 pinned 的容器/进程，需要再留余量。
- 一句话结论：**这两个选项不改变"AI 智力"，只改变"多轮对话有多快"**——即 prefix
  复用命中率与首次 token 延迟。要更快就尽量给 State 留槽；要更省就设 0。

### 方案 4（要完整默认行为）：原生 Linux

裸机 Linux 下默认 `8 slots + 8 GiB Host KV` 可正常工作，`--ulimit memlock` 才真正生效。
这是想要上游默认配置的唯一稳定解。

### 不推荐 / 无效

- 只调 `--host-state-slots` 而保留默认 `--host-kv-mib 8192` → 会在 `pinning host KV` 阶段失败
  （[`program_impl.h:911`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911)）。
- 调大 `.wslconfig memory=`、`--ulimit memlock`、`--ipc=host`、`--shm-size`、Docker Desktop MemoryMiB
  → 均已在 [`test_pin_host.md`](./test_pin_host.md) 中实测排除。
- 把大分配拆成多个小块同时持有 → 总量仍受限，无效。

## C. 验证步骤

```powershell
# 1) 先量出本机当前可用 pinned 预算
.\test_pin_host.ps1 -Budget            # 期望 ≈ 1136–1152 MiB

# 2) 用方案 2 或 3 启动，观察日志
#    期望：pinning host state | <≤ 1 GiB，或整行不出现>
#          pinning host KV    | <≤ 1 GiB，或整行不出现>
#    不再出现 FATAL cudaMallocHost failed
```

## D. 建议（可选，需改代码，本次未做）

若希望开箱即用，可考虑：

1. 在 WSL2 / 容器环境下把 [`kDefaultHostStateSlots`](./ninfer-4090/include/ninfer/types.h#L27)、[`kDefaultHostKvCapacityBytes`](./ninfer-4090/include/ninfer/types.h#L28) 降为
   WSL2 友好值，或启动时探测 `cudaMallocHost` 可用量后再定容；
2. 让 [`HostStatePool`](./ninfer-4090/src/targets/qwen3_6/impl/state/state_image.cpp#L211) / [`HostKVArena`](./ninfer-4090/src/core/host_kv_arena.cpp#L211) 的 pinned 申请失败时 **降级为 device-only 而非 FATAL 退出**，
   并打印一条 WARN 说明已关闭 host 复用能力。

## E. 每个方案对应的源码位置与源码版本

### 核对所用源码版本（ninfer-4090）

| 项 | 值 |
| --- | --- |
| 仓库 | `ninfer-4090`（`h:\WorkShopAI\ninfer-4090-docker-compose\ninfer-4090`） |
| commit | `aeeba414459d5d6989d57d8487c9d7a2f54bddd3` |
| 短 hash | `aeeba414` |
| `git describe` | `v0.6.0-rtx3090-401-gaeeba414` |
| 提交时间 | 2026-09-23 14:51:55 +0200 |
| 提交标题 | `docs(ledger): record the per-item vision budget fix` |

> 下文所有行号都基于上面的 `aeeba414`。若后续 rebase / 合并，请以 `git log -1` 重新对齐。
> 链接均为**相对路径**（`ninfer-4090/` 是 submodule），在 VS Code Markdown 预览里点击即跳到对应文件的对应行；
> 在 GitHub 上需在本文件所在的仓库视图中浏览。

### 公共（所有方案都相关）的源码点

| 环节 | 源码位置 |
| --- | --- |
| pinned 原语 `PinnedHostBuffer` → `cudaMallocHost`（失败即 `throw` → FATAL） | [`src/core/arena.cu:244-256`](./ninfer-4090/src/core/arena.cu#L244-L256) |
| 默认容量常量 `kDefaultHostStateSlots=8`、`kDefaultHostKvCapacityBytes=8 GiB` | [`include/ninfer/types.h:27-28`](./ninfer-4090/include/ninfer/types.h#L27-L28) |
| 默认值挂到 `ContextCacheOptions` 字段 | [`include/ninfer/types.h:153-154`](./ninfer-4090/include/ninfer/types.h#L153-L154) |
| Host State pin 阶段（`StartupPhase::HostStatePin`） | [`src/targets/qwen3_6/impl/runtime/program_impl.h:851-859`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851-L859) |
| `HostStatePool` 的 pinned 后端 `backing_.emplace(bytes)` | [`src/targets/qwen3_6/impl/state/state_image.cpp:211-225`](./ninfer-4090/src/targets/qwen3_6/impl/state/state_image.cpp#L211-L225) |
| Host KV pin 阶段（`StartupPhase::HostKvPin`） | [`src/targets/qwen3_6/impl/runtime/program_impl.h:911-928`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911-L928) |
| `HostKVArena` 的 pinned 后端 `backing_.emplace(capacity_bytes_)` | [`src/core/host_kv_arena.cpp:231`](./ninfer-4090/src/core/host_kv_arena.cpp#L231) |
| 单份 StateImage ≈ 147 MiB（Qwen3.8-27B） | [`docs/turn-checkpoint-ring.md`](./ninfer-4090/docs/turn-checkpoint-ring.md)（"What one checkpoint holds"） |
| 文档默认值 `--host-state-slots 8` / `--host-kv-mib 8192` | [`docs/serving.md:840-841`](./ninfer-4090/docs/serving.md#L840-L841) |

### 方案 1：`--no-prefix-reuse`（零 pinned，关闭 context cache）

| 作用 | 源码位置 |
| --- | --- |
| 解析该开关 → `options.allow_prefix_reuse = false` | [`src/serve/serve_options.cpp:336-337`](./ninfer-4090/src/serve/serve_options.cpp#L336-L337) |
| **联动清零** `context_cache.enabled=false`、`host_state_slots=0`、`host_kv_capacity_bytes=0`、`auto_long_anchors=0` | [`src/serve/serve_options.cpp:381-390`](./ninfer-4090/src/serve/serve_options.cpp#L381-L390)（赋值在 [`386-389`](./ninfer-4090/src/serve/serve_options.cpp#L386-L389)） |
| 与显式 cache 容量参数 **互斥**（`--no-prefix-reuse cannot be combined with context-cache capacity options`） | [`src/serve/serve_options.cpp:382-385`](./ninfer-4090/src/serve/serve_options.cpp#L382-L385) |
| `--help` / usage 文本 | [`src/serve/serve_options.cpp:91`](./ninfer-4090/src/serve/serve_options.cpp#L91)、[`:132`](./ninfer-4090/src/serve/serve_options.cpp#L132) |
| 由于容量为 0，两个 pin 阶段被跳过（`if (... != 0)` 守卫） | [`program_impl.h:851`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851)、[`program_impl.h:911`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911) |
| Engine 归一化侧的另一条清零路径（`EnginePurpose::CausalScoring`） | [`src/runtime/engine/engine.cpp:47`](./ninfer-4090/src/runtime/engine/engine.cpp#L47)；`!cache.enabled` 分支 [`58-72`](./ninfer-4090/src/runtime/engine/engine.cpp#L58-L72)（`cache.host_state_slots = 0` 在 [`67`](./ninfer-4090/src/runtime/engine/engine.cpp#L67)） |

### 方案 2：`--host-state-slots 0 --host-kv-mib 0`（保留 context cache，纯 Device）

| 作用 | 源码位置 |
| --- | --- |
| 解析 `--host-state-slots` → `context_cache.host_state_slots`（并置 `context_capacity_explicit = true`） | [`src/serve/serve_options.cpp:251-254`](./ninfer-4090/src/serve/serve_options.cpp#L251-L254)（赋值 [`252-253`](./ninfer-4090/src/serve/serve_options.cpp#L252-L253)，标记 [`254`](./ninfer-4090/src/serve/serve_options.cpp#L254)） |
| 解析 `--host-kv-mib` → `context_cache.host_kv_capacity_bytes` | [`src/serve/serve_options.cpp:255-261`](./ninfer-4090/src/serve/serve_options.cpp#L255-L261)（解析 [`256`](./ninfer-4090/src/serve/serve_options.cpp#L256)，上界检查 [`257-259`](./ninfer-4090/src/serve/serve_options.cpp#L257-L259)，赋值 [`260`](./ninfer-4090/src/serve/serve_options.cpp#L260)，标记 [`261`](./ninfer-4090/src/serve/serve_options.cpp#L261)） |
| 值 `0` 使 Host State pin 被跳过（`if (plan.context_cache.host_state_slots != 0)`） | [`src/targets/qwen3_6/impl/runtime/program_impl.h:851`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851) |
| 值 `0` 使 Host KV pin 被跳过（`if (plan.context_cache.host_kv_capacity_bytes != 0)`） | [`src/targets/qwen3_6/impl/runtime/program_impl.h:911`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911) |
| `0` 是合法值（`--host-state-slots` 用 `parse_nonnegative_int`；`--host-kv-mib` 用 `parse_u64` 且只有上界检查，无 `> 0` 断言） | [`src/serve/serve_options.cpp:253`](./ninfer-4090/src/serve/serve_options.cpp#L253)、[`:256-259`](./ninfer-4090/src/serve/serve_options.cpp#L256-L259) |
| Engine 归一化只补 `device_state_slots` / `max_private_*` 等，**不覆盖** `host_state_slots` → 显式 `0` 会保留 | [`src/runtime/engine/engine.cpp:74-102`](./ninfer-4090/src/runtime/engine/engine.cpp#L74-L102)（启用分支） |

### 方案 3：部分 pinned（`--host-state-slots 1 --host-kv-mib 768` 等）

源码点与**方案 2 完全相同**（同为 [`serve_options.cpp:251-261`](./ninfer-4090/src/serve/serve_options.cpp#L251-L261) 解析 + [`program_impl.h:851`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851)/[`911`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911) 守卫）。
区别只在数值满足预算不等式，参数来源为 [`include/ninfer/types.h:27-28`](./ninfer-4090/include/ninfer/types.h#L27-L28) 的两个默认常量：

$$\text{host\_state\_slots} \times 147\ \text{MiB} + \text{--host-kv-mib} \le \approx 1100\ \text{MiB}$$

### 方案 4：原生 Linux（用上游默认 `8 + 8 GiB`）

- **不涉及 ninfer-4090 源码改动**：同一份 `aeeba414` 二进制在裸机 Linux 上按默认值即可启动。
- 限制只来自 WSL2 驱动侧的全局 pinned 池（本机 KMD 617.14 / `dxgkrnl`），
  实测与根因见 [`test_pin_host.md`](./test_pin_host.md)（第 3、5 节）。
- 相关上游依据：NVIDIA《CUDA on WSL User Guide》§5.1 Known Limitations。

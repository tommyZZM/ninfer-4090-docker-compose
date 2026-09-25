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

# Addendum (2026-09-25): root cause and fixes located from the ninfer-4090 source

## A. Root cause (code-level evidence)

`pinning host state | 1.15 GiB` is the pinned backing allocation of `HostStatePool`:

| Stage | Code location | Notes |
| --- | --- | --- |
| pinned primitive | [`src/core/arena.cu:244`](./ninfer-4090/src/core/arena.cu#L244) | `PinnedHostBuffer` → `cudaMallocHost`; on failure it throws `cudaMallocHost failed: ... out of memory`, matching the FATAL line in the log |
| default capacities | [`include/ninfer/types.h:153-154`](./ninfer-4090/include/ninfer/types.h#L153-L154) | `host_state_slots = 8`, `host_kv_capacity_bytes = 8 GiB` |
| Host State allocation | [`src/targets/qwen3_6/impl/runtime/program_impl.h:851-859`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851-L859) | `host_state_bytes = image_bytes × host_state_slots` → `HostStatePool` |
| Host State backing | [`src/targets/qwen3_6/impl/state/state_image.cpp:212-225`](./ninfer-4090/src/targets/qwen3_6/impl/state/state_image.cpp#L212-L225) | `backing_.emplace(bytes)` = `cudaMallocHost` |
| single-slot size | [`docs/turn-checkpoint-ring.md`](./ninfer-4090/docs/turn-checkpoint-ring.md) | one StateImage on Qwen3.8-27B ≈ 147 MiB |
| Host KV allocation | [`src/targets/qwen3_6/impl/runtime/program_impl.h:911-928`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911-L928) | `HostKVArena(capacity)` → `backing_.emplace` at [`src/core/host_kv_arena.cpp:231`](./ninfer-4090/src/core/host_kv_arena.cpp#L231) = another `cudaMallocHost` |
| parsing entry points | [`src/serve/serve_options.cpp:252`](./ninfer-4090/src/serve/serve_options.cpp#L252), [`:260`](./ninfer-4090/src/serve/serve_options.cpp#L260) | `--host-state-slots` / `--host-kv-mib` |
| documented defaults | [`docs/serving.md:840-841`](./ninfer-4090/docs/serving.md#L840-L841) | `--host-state-slots` defaults to `8`; `--host-kv-mib` defaults to `8192` |

Numeric cross-check:

```text
host state pinned = 8 × 147.2 MiB = 1177.6 MiB = 1.148 GiB  (exactly the 1.15 GiB in the log)
measured WSL2 global pinned pool ceiling = [1136, 1152) MiB
=> 1.148 GiB is just over the line, so cudaMallocHost OOM is inevitable
```

Conclusion: this is **default pinned capacity > WSL2 global pool ceiling**. It is not a misuse of
`cudaMallocHost`, and not a `.wslconfig` / `memlock` / `shm` / `ipc` configuration problem (this
matches the measurements in [`test_pin_host.md`](./test_pin_host.md) in this same directory).
`ninfer-serve` has no degrade/fallback path at this stage: a failed allocation `throw`s → FATAL exit.

### Related risk: the default 8 GiB Host KV is "the next landmine"

Even if host state is brought under the ceiling, the immediately following `HostKvPin` stage will
request pinned memory again using the default `host_kv_capacity_bytes = 8 GiB`, which must fail
for the same reason.
**`--host-state-slots` and `--host-kv-mib` are two independent pinned capacities and must be lowered together.**

(Separately: the weight staging in [`src/artifact/materializer.cpp`](./ninfer-4090/src/artifact/materializer.cpp#L183-L266)
uses 64 MiB × ≤4 pinned, but it calls `slots.clear()` before `pinning host state`, so it does not
consume this budget; `PagedKVCache::host_shadow_`
([`src/core/paged_kv_cache.h:377`](./ninfer-4090/src/core/paged_kv_cache.h#L377)) is only on the MB scale.)

### Relationship to the 4090D

Pinned host memory lives on the Windows-side driver (KMD 617.14 / `dxgkrnl`) and the WSL2 VM host
memory path, and is independent of the GPU model (4090 / 4090D / 5090). So the difference between
the 4090D and the upstream 4090 **does not** move this boundary; the ceiling is determined solely by
the size of the WSL2 driver's global pool.

### What `--host-state-slots` and `--host-kv-mib` are actually for

Both options serve exactly one purpose: the **demotion capability of the context cache** — moving
inactive checkpoints out of VRAM into pinned host memory so they survive Device pressure instead of
being discarded. They **do not** increase concurrency, do not raise any single request's context
ceiling, and take no part in the execution path of active requests.

| Option | What it holds | Unit | Default | Source |
| --- | --- | --- | --- | --- |
| `--host-state-slots` | **complete StateImage replicas** = the hybrid model's GDN / linear-attention recurrent state + conv state + boundary hidden. This is the "continuable session state"; each replica is about **147 MiB** (Qwen3.8-27B) | slot count (× per-image `image_bytes`) | `8` | [`include/ninfer/types.h:153`](./ninfer-4090/include/ninfer/types.h#L153) → [`state_image.cpp:212-225`](./ninfer-4090/src/targets/qwen3_6/impl/state/state_image.cpp#L212-L225) |
| `--host-kv-mib` | **host replicas of typed packed KV pages**, allocated from a single arena **shared by** the Main Text KV pool and the selected backend (MTP / DFlash) pool, carved into physical page extents | MiB (byte capacity) | `8192` (8 GiB) | [`include/ninfer/types.h:154`](./ninfer-4090/include/ninfer/types.h#L154) → [`host_kv_arena.cpp:231`](./ninfer-4090/src/core/host_kv_arena.cpp#L231) |

#### Why they are two independent capacities

State and KV are **independently placed** resources
(`docs/maintainer/resource-scheduling-and-context-cache.md` §5.1/§5.2): one checkpoint may have its
State on Host and its KV on Device, or the reverse. `Host StateImages` and `Host KV bytes` are
therefore billed separately and are **not interchangeable**:

- A StateImage is a **complete migration unit** — it is never partially demoted, and one replica
  consumes one whole slot.
- Host KV is a **byte pool allocated in extents**, shared by Main and Backend, sized as
  `page_stride × pages`.

#### What actually consumes them

| Consumer | Source location |
| --- | --- |
| startup allocation of the pinned StateImage pool by slot count | [`program_impl.h:851-859`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851-L859) |
| startup allocation of the pinned Host KV arena by bytes | [`program_impl.h:911-928`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911-L928) |
| pressure planning **demoting Device-only State to Host** (`DemoteSharedToHost` / `DemoteEndpointToHost` / `DemoteRewriteToHost`) | [`program_impl.h:1468-1479`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L1468-L1479), [`1663-1704`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L1663-L1704) |
| pressure planning **demoting KV pages to Host** (`PressureKVDecisionKind::DemoteToHost`) | [`program_impl.h:1492-1501`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L1492-L1501), [`2669-2681`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L2669-L2681) |
| the three placement states `DeviceOnly` / `HostOnly` / `Both` | [`program_impl.h:2214-2220`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L2214-L2220) |
| Host KV extent ownership and page membership | [`host_kv_extent_store.h`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/host_kv_extent_store.h) |
| architecture definition ("Host State and Host KV are billed independently") | [`docs/maintainer/resource-scheduling-and-context-cache.md`](./ninfer-4090/docs/maintainer/resource-scheduling-and-context-cache.md) §11 |
| user-facing description ("Host KV is shared by Main and the selected Backend pool and is consumed in physical page extents") | [`docs/serving.md:870-875`](./ninfer-4090/docs/serving.md#L870-L875) |

#### How this relates to the failure

Because they are **independent pinned capacities** with defaults of `8 × 147 MiB` and `8 GiB`:

```text
Host State  8 × 147.2 MiB = 1.148 GiB   <- hits the ceiling first; the log stops at "pinning host state"
Host KV     8 GiB         = 8192 MiB    <- would still hit it even after the former is lowered
```

**Both must be lowered together** for the server to start. Setting them to `0` means giving up the
Host demotion capability: checkpoints can then only stay on Device or be evicted/discarded —
**the execution and correctness of active requests are unaffected**; you only lose the cache's
ability to survive Device pressure (see the last paragraph of §11 in
`docs/maintainer/...md`: with the context cache disabled, root-only semantics apply and Device/Host
checkpoint capacity is zero).

## B. Fixes (ordered by recommendation)

Let \(B \approx 1100\) MiB be the usable pinned budget (leaving ~50 MiB for other users in the same
VM). The constraint is:

\[
\text{host\_state\_slots} \times 147\ \text{MiB} + \text{--host-kv-mib} \le B
\]

### Option 1 (most robust, zero pinned): `--no-prefix-reuse`

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

- Effect: [`src/serve/serve_options.cpp:385-388`](./ninfer-4090/src/serve/serve_options.cpp#L385-L388)
  sets `context_cache.enabled = false`, `host_state_slots = 0`, `host_kv_capacity_bytes = 0`, so both
  pin stages are skipped entirely.
- Cost: the context cache is disabled (no cross-request prefix reuse, no continuation retention).
- Caveat: `--no-prefix-reuse` **cannot be combined with any explicit context-cache capacity option**
  (including zero-valued ones); see [`docs/serving.md`](./ninfer-4090/docs/serving.md).

### Option 2 (recommended compromise): keep the context cache, but run Device-only with no pinned memory

```bash
  ... --host-state-slots 0 --host-kv-mib 0
```

- Effect: the context cache stays enabled and device checkpoints remain available; only inactive
  continuations can no longer be demoted to the Host. Both pin stages are skipped because the
  capacities are 0.
- Suited to single-concurrency multi-turn chat: prefix reuse still works, there is just no pinned
  host fallback.

### Option 3 (partial pinned fallback needed): squeeze the total into the budget

```bash
  ... --host-state-slots 1 --host-kv-mib 768     # ≈ 147 + 768 = 915 MiB
  # or
  ... --host-state-slots 0 --host-kv-mib 1024    # = 1024 MiB
```

- Cross-check: `1 × 147.2 + 768 = 915 MiB ≤ 1100 MiB`, safe.
- Caveat: the WSL2 pool is **VM-global**. If other containers/processes on the same machine also
  request pinned memory, leave additional headroom.

### Option 4 (want full default behavior): native Linux

On bare-metal Linux the defaults `8 slots + 8 GiB Host KV` work normally, and `--ulimit memlock`
actually takes effect. This is the only stable way to keep the upstream default configuration.

### Not recommended / ineffective

- Tuning only `--host-state-slots` while keeping the default `--host-kv-mib 8192` → it will fail at
  the `pinning host KV` stage
  ([`program_impl.h:911`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911)).
- Raising `.wslconfig memory=`, `--ulimit memlock`, `--ipc=host`, `--shm-size`, or Docker Desktop
  MemoryMiB → all of these were already measured and ruled out in
  [`test_pin_host.md`](./test_pin_host.md).
- Splitting one large allocation into many smaller ones held simultaneously → the total is still
  bounded, so this does not help.

## C. Verification steps

```powershell
# 1) First measure the currently usable pinned budget on this machine
.\test_pin_host.ps1 -Budget            # expect ≈ 1136–1152 MiB

# 2) Start with Option 2 or Option 3 and watch the log
#    expect: pinning host state | <≤ 1 GiB, or the line does not appear at all>
#            pinning host KV    | <≤ 1 GiB, or the line does not appear at all>
#    no more FATAL cudaMallocHost failed
```

## D. Suggestions (optional, would require code changes — not done here)

For out-of-the-box behavior, consider:

1. Under WSL2 / container environments, lower
   [`kDefaultHostStateSlots`](./ninfer-4090/include/ninfer/types.h#L27) and
   [`kDefaultHostKvCapacityBytes`](./ninfer-4090/include/ninfer/types.h#L28) to WSL2-friendly values,
   or probe the available `cudaMallocHost` capacity at startup before sizing them;
2. Make a failed pinned allocation in
   [`HostStatePool`](./ninfer-4090/src/targets/qwen3_6/impl/state/state_image.cpp#L211) /
   [`HostKVArena`](./ninfer-4090/src/core/host_kv_arena.cpp#L211)
   **degrade to device-only instead of exiting FATAL**, and print a WARN stating that host
   reuse capability has been disabled.

## E. Source locations and source revision per option

### Source revision used for verification (ninfer-4090)

| Item | Value |
| --- | --- |
| repository | `ninfer-4090` (`h:\WorkShopAI\ninfer-4090-docker-compose\ninfer-4090`) |
| commit | `aeeba414459d5d6989d57d8487c9d7a2f54bddd3` |
| short hash | `aeeba414` |
| `git describe` | `v0.6.0-rtx3090-401-gaeeba414` |
| commit date | 2026-09-23 14:51:55 +0200 |
| commit subject | `docs(ledger): record the per-item vision budget fix` |

> Every line number below is based on `aeeba414` above. After a rebase / merge, re-align with
> `git log -1`.
> All links are **relative paths** (`ninfer-4090/` is a submodule). In the VS Code Markdown preview,
> clicking one jumps straight to the corresponding line of the corresponding file. On GitHub, this
> file must be browsed from the repository view that contains it.

### Shared source points (relevant to every option)

| Stage | Source location |
| --- | --- |
| pinned primitive `PinnedHostBuffer` → `cudaMallocHost` (throws on failure → FATAL) | [`src/core/arena.cu:244-256`](./ninfer-4090/src/core/arena.cu#L244-L256) |
| default capacity constants `kDefaultHostStateSlots=8`, `kDefaultHostKvCapacityBytes=8 GiB` | [`include/ninfer/types.h:27-28`](./ninfer-4090/include/ninfer/types.h#L27-L28) |
| defaults bound to the `ContextCacheOptions` fields | [`include/ninfer/types.h:153-154`](./ninfer-4090/include/ninfer/types.h#L153-L154) |
| Host State pin stage (`StartupPhase::HostStatePin`) | [`src/targets/qwen3_6/impl/runtime/program_impl.h:851-859`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851-L859) |
| `HostStatePool` pinned backing `backing_.emplace(bytes)` | [`src/targets/qwen3_6/impl/state/state_image.cpp:211-225`](./ninfer-4090/src/targets/qwen3_6/impl/state/state_image.cpp#L211-L225) |
| Host KV pin stage (`StartupPhase::HostKvPin`) | [`src/targets/qwen3_6/impl/runtime/program_impl.h:911-928`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911-L928) |
| `HostKVArena` pinned backing `backing_.emplace(capacity_bytes_)` | [`src/core/host_kv_arena.cpp:231`](./ninfer-4090/src/core/host_kv_arena.cpp#L231) |
| one StateImage ≈ 147 MiB (Qwen3.8-27B) | [`docs/turn-checkpoint-ring.md`](./ninfer-4090/docs/turn-checkpoint-ring.md) ("What one checkpoint holds") |
| documented defaults `--host-state-slots 8` / `--host-kv-mib 8192` | [`docs/serving.md:840-841`](./ninfer-4090/docs/serving.md#L840-L841) |

### Option 1: `--no-prefix-reuse` (zero pinned, context cache disabled)

| Effect | Source location |
| --- | --- |
| parses the flag → `options.allow_prefix_reuse = false` | [`src/serve/serve_options.cpp:336-337`](./ninfer-4090/src/serve/serve_options.cpp#L336-L337) |
| **cascading zeroing** of `context_cache.enabled=false`, `host_state_slots=0`, `host_kv_capacity_bytes=0`, `auto_long_anchors=0` | [`src/serve/serve_options.cpp:381-390`](./ninfer-4090/src/serve/serve_options.cpp#L381-L390) (assignments at [`386-389`](./ninfer-4090/src/serve/serve_options.cpp#L386-L389)) |
| mutually exclusive with explicit cache capacity options (`--no-prefix-reuse cannot be combined with context-cache capacity options`) | [`src/serve/serve_options.cpp:382-385`](./ninfer-4090/src/serve/serve_options.cpp#L382-L385) |
| `--help` / usage text | [`src/serve/serve_options.cpp:91`](./ninfer-4090/src/serve/serve_options.cpp#L91), [`:132`](./ninfer-4090/src/serve/serve_options.cpp#L132) |
| both pin stages are skipped because the capacities are 0 (the `if (... != 0)` guards) | [`program_impl.h:851`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851), [`program_impl.h:911`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911) |
| the other zeroing path on the Engine normalization side (`EnginePurpose::CausalScoring`) | [`src/runtime/engine/engine.cpp:47`](./ninfer-4090/src/runtime/engine/engine.cpp#L47); the `!cache.enabled` branch [`58-72`](./ninfer-4090/src/runtime/engine/engine.cpp#L58-L72) (`cache.host_state_slots = 0` at [`67`](./ninfer-4090/src/runtime/engine/engine.cpp#L67)) |

### Option 2: `--host-state-slots 0 --host-kv-mib 0` (keep the context cache, Device-only)

| Effect | Source location |
| --- | --- |
| parses `--host-state-slots` → `context_cache.host_state_slots` (and sets `context_capacity_explicit = true`) | [`src/serve/serve_options.cpp:251-254`](./ninfer-4090/src/serve/serve_options.cpp#L251-L254) (assignment [`252-253`](./ninfer-4090/src/serve/serve_options.cpp#L252-L253), flag [`254`](./ninfer-4090/src/serve/serve_options.cpp#L254)) |
| parses `--host-kv-mib` → `context_cache.host_kv_capacity_bytes` | [`src/serve/serve_options.cpp:255-261`](./ninfer-4090/src/serve/serve_options.cpp#L255-L261) (parse [`256`](./ninfer-4090/src/serve/serve_options.cpp#L256), upper-bound check [`257-259`](./ninfer-4090/src/serve/serve_options.cpp#L257-L259), assignment [`260`](./ninfer-4090/src/serve/serve_options.cpp#L260), flag [`261`](./ninfer-4090/src/serve/serve_options.cpp#L261)) |
| value `0` skips the Host State pin (`if (plan.context_cache.host_state_slots != 0)`) | [`src/targets/qwen3_6/impl/runtime/program_impl.h:851`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851) |
| value `0` skips the Host KV pin (`if (plan.context_cache.host_kv_capacity_bytes != 0)`) | [`src/targets/qwen3_6/impl/runtime/program_impl.h:911`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911) |
| `0` is a legal value (`--host-state-slots` uses `parse_nonnegative_int`; `--host-kv-mib` uses `parse_u64` with only an upper-bound check, no `> 0` assertion) | [`src/serve/serve_options.cpp:253`](./ninfer-4090/src/serve/serve_options.cpp#L253), [`:256-259`](./ninfer-4090/src/serve/serve_options.cpp#L256-L259) |
| Engine normalization only fills in `device_state_slots` / `max_private_*` etc. and does **not** overwrite `host_state_slots`, so an explicit `0` is preserved | [`src/runtime/engine/engine.cpp:74-102`](./ninfer-4090/src/runtime/engine/engine.cpp#L74-L102) (enabled branch) |

### Option 3: partial pinned (`--host-state-slots 1 --host-kv-mib 768`, etc.)

The source points are **identical to Option 2** (the same parsing at
[`serve_options.cpp:251-261`](./ninfer-4090/src/serve/serve_options.cpp#L251-L261) plus the guards at
[`program_impl.h:851`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L851)/[`911`](./ninfer-4090/src/targets/qwen3_6/impl/runtime/program_impl.h#L911)).
The only difference is that the values satisfy the budget inequality; the parameters originate from
the two default constants in [`include/ninfer/types.h:27-28`](./ninfer-4090/include/ninfer/types.h#L27-L28):

$$\text{host\_state\_slots} \times 147\ \text{MiB} + \text{--host-kv-mib} \le \approx 1100\ \text{MiB}$$

### Option 4: native Linux (uses the upstream defaults `8 + 8 GiB`)

- **No ninfer-4090 source change is involved**: the same `aeeba414` binary starts with the default
  values on bare-metal Linux.
- The limitation comes solely from the WSL2 driver-side global pinned pool (KMD 617.14 / `dxgkrnl`
  on this machine); measurements and root cause are in
  [`test_pin_host.md`](./test_pin_host.md) (sections 3 and 5).
- Related upstream reference: NVIDIA, *CUDA on WSL User Guide*, §5.1 Known Limitations.

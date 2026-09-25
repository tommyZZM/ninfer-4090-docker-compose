# 调试方式通过 .\test_pin_host.ps1 运行
#
#   .\test_pin_host.ps1 -SizeGb 1                 # 单个 1 GiB 锁定内存分配
#   .\test_pin_host.ps1 -SizeGb 0.5 -Repeat 4     # 4 次 x 0.5 GiB（总量 2 GiB）
#   .\test_pin_host.ps1 -Probe -SizeGb 4          # 二分查找单次分配上限
#   .\test_pin_host.ps1 -Budget                   # 逐块探测 pinned 总量上限
#   .\test_pin_host.ps1 -SizeGb 1 -Cycles 4       # 反复申请/释放 1 GiB（环形缓冲可行性）
#   .\test_pin_host.ps1 -SizeGb 2 -Mode register  # 用 cudaHostRegister 代替 cudaMallocHost
#
# 也可以直接进容器运行：
#   python3 test_pin_host.py --size-gb 2 --repeat 1
#
# 环境变量 PIN_SIZE_GB / PIN_REPEAT / PIN_MODE / PIN_CYCLES 同样生效（供 docker run -e 使用）。
import argparse
import ctypes
import glob
import mmap
import os
import platform
import resource
import sys

import psutil

DEFAULTS = {
    "size_gb": float(os.environ.get("PIN_SIZE_GB", 2)),
    "repeat": int(os.environ.get("PIN_REPEAT", 1)),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Probe CUDA pinned host memory limits (cudaMallocHost / cudaHostRegister)."
    )
    parser.add_argument(
        "--size-gb",
        type=float,
        default=DEFAULTS["size_gb"],
        help="size of each pinned allocation in GiB (default: %(default)s)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULTS["repeat"],
        help="number of pinned allocations held simultaneously (default: %(default)s)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="binary-search the largest single pinned allocation that succeeds",
    )
    parser.add_argument(
        "--budget",
        action="store_true",
        help="hold 16 MiB chunks until failure and report the total pinned budget",
    )
    parser.add_argument(
        "--mode",
        choices=("malloc", "register"),
        default=os.environ.get("PIN_MODE", "malloc"),
        help="malloc = cudaMallocHost; register = cudaHostRegister over pageable memory",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=int(os.environ.get("PIN_CYCLES", 1)),
        help="alloc/free rounds per buffer; >1 proves buffers can be recycled (default: %(default)s)",
    )
    parser.add_argument(
        "--libcudart",
        default=os.environ.get("LIBCUDART_PATH"),
        help="explicit path to libcudart.so",
    )
    args = parser.parse_args(argv)
    if args.size_gb <= 0:
        parser.error("--size-gb must be > 0")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.cycles < 1:
        parser.error("--cycles must be >= 1")
    return args


def format_bytes(b):
    if b == resource.RLIM_INFINITY:
        return "Unlimited"
    return f"{b / (1024**3):.2f} GiB ({b} bytes)"


def read_first(path, prefixes):
    """Return the value part of the first line in `path` starting with a prefix."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for prefix in prefixes:
                    if line.startswith(prefix):
                        return line.split(":", 1)[-1].strip()
    except OSError:
        return None
    return None


def report_environment():
    mem = psutil.virtual_memory()
    print(f"Python:                 {platform.python_version()} ({sys.executable})")
    print(f"VM Memory (Total):      {mem.total / (1024**3):.2f} GiB")
    print(f"VM Memory (Available):  {mem.available / (1024**3):.2f} GiB")

    for label, path in (
        ("cgroup v2 memory", "/sys/fs/cgroup/memory.max"),
        ("cgroup v1 memory", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        value = read_first(path, [""])
        if value is not None:
            print(f"{label}: {value}")

    meminfo = {
        key: read_first("/proc/meminfo", [key])
        for key in ("MemTotal", "MemAvailable", "CommitLimit", "Committed_AS")
    }
    print(
        "MemInfo:                "
        + ", ".join(f"{k}={v}" for k, v in meminfo.items() if v is not None)
    )

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    print("=== RLIMIT_MEMLOCK ===")
    print(f"Soft Limit:             {format_bytes(soft_limit)}")
    print(f"Hard Limit:             {format_bytes(hard_limit)}")
    print("=" * 60)


def load_cudart(explicit_path=None):
    candidates = [explicit_path] if explicit_path else []
    candidates += [
        "/opt/conda/lib/libcudart.so.12",
        "/opt/conda/lib/libcudart.so",
        "libcudart.so.12",
        "libcudart.so",
    ]
    candidates += sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"))

    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        print(f"libcudart:              {candidate}")
        lib.cudaMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMallocHost.restype = ctypes.c_int
        lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        lib.cudaFreeHost.restype = ctypes.c_int
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        return lib
    raise RuntimeError("unable to load libcudart:\n  " + "\n  ".join(errors))


class Pinner:
    """Pins host memory via cudaMallocHost (malloc) or cudaHostRegister (register)."""

    def __init__(self, lib, mode):
        self.lib = lib
        self.mode = mode
        if mode == "register":
            lib.cudaHostRegister.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            lib.cudaHostRegister.restype = ctypes.c_int
            lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            lib.cudaHostUnregister.restype = ctypes.c_int

    def acquire(self, size_bytes):
        """Pin `size_bytes`. Returns (error_code, handle); handle is None on failure."""
        if self.mode == "malloc":
            ptr = ctypes.c_void_p()
            err = self.lib.cudaMallocHost(ctypes.byref(ptr), ctypes.c_size_t(size_bytes))
            if err != 0:
                return err, None
            return 0, (ptr, None)

        # cudaHostRegister requires page-aligned pageable memory: anonymous mmap gives that.
        buffer = mmap.mmap(-1, size_bytes)
        address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        err = self.lib.cudaHostRegister(
            ctypes.c_void_p(address), ctypes.c_size_t(size_bytes), 0
        )
        if err != 0:
            buffer.close()
            return err, None
        return 0, (ctypes.c_void_p(address), buffer)

    def release(self, handle):
        pointer, buffer = handle
        if self.mode == "malloc":
            self.lib.cudaFreeHost(pointer)
        else:
            self.lib.cudaHostUnregister(pointer)
        if buffer is not None:
            buffer.close()


def err_text(lib, err):
    message = lib.cudaGetErrorString(err)
    return message.decode() if message else f"error code {err}"


def run_single(pinner, size_gb, repeat, cycles):
    size_bytes = int(size_gb * 1024**3)
    print(
        f"--- {pinner.mode}: {size_gb:g} GiB x {repeat} held together, cycles={cycles} ---"
    )

    peak = 0
    ok_all = True
    for cycle in range(cycles):
        held = []
        for index in range(repeat):
            err, handle = pinner.acquire(size_bytes)
            if err != 0:
                print(
                    f"FAILED: {pinner.mode} block #{index + 1} of {repeat} "
                    f"({size_gb:g} GiB) -> {err_text(pinner.lib, err)} (code {err})"
                )
                ok_all = False
                break
            held.append(handle)
            peak = max(peak, (index + 1) * size_gb)
            print(
                f"  ok #{index + 1}/{repeat}: {size_gb:g} GiB "
                f"(held total {(index + 1) * size_gb:.2f} GiB)"
            )
        for handle in held:
            pinner.release(handle)
        if not ok_all:
            break
        if cycles > 1:
            print(f"  cycle {cycle + 1}/{cycles}: acquired and released all blocks")

    if ok_all:
        print(
            f"SUCCESS: {pinner.mode} held up to {peak:.2f} GiB "
            f"({size_gb:g} GiB x {repeat}, {cycles} cycle(s))"
        )
    else:
        print(f"Peak held before failing: {peak:.2f} GiB")
    return ok_all


CHUNK_MIB = 16


def run_probe(pinner, upper_gb):
    """Binary-search the largest single allocation that succeeds."""
    print(f"--- probing largest single {pinner.mode} block up to {upper_gb:g} GiB ---")

    def attempt(size_gb):
        err, handle = pinner.acquire(int(size_gb * 1024**3))
        if err == 0:
            pinner.release(handle)
        return err == 0

    if attempt(upper_gb):
        print(f"upper bound {upper_gb:g} GiB succeeded; raise the bound to find the ceiling.")
        return True

    low, high = 0.0, upper_gb
    for _ in range(12):
        mid = (low + high) / 2
        if mid - low < 0.0625:
            break
        success = attempt(mid)
        print(f"  {mid:.3f} GiB -> {'ok' if success else 'fail'}")
        if success:
            low = mid
        else:
            high = mid

    print(
        f"largest single {pinner.mode} block that works: ~{low:.3f} GiB "
        f"(ceiling between {low:.3f} and {high:.3f} GiB)"
    )
    return low > 0


def run_budget(pinner, max_gib=8):
    """Hold CHUNK_MIB blocks until failure to measure the total pinned budget."""
    chunk_bytes = CHUNK_MIB * 1024 * 1024
    limit = int(max_gib * 1024**3)
    print(f"--- measuring total {pinner.mode} budget with {CHUNK_MIB} MiB blocks ---")

    held = []
    total_mib = 0
    while (total_mib + CHUNK_MIB) * 1024 * 1024 <= limit:
        err, handle = pinner.acquire(chunk_bytes)
        if err != 0:
            print(
                f"FAILED at block {len(held) + 1} -> {err_text(pinner.lib, err)} "
                f"(code {err}); held {total_mib} MiB"
            )
            break
        held.append(handle)
        total_mib += CHUNK_MIB

    print(
        f"total {pinner.mode} budget: {total_mib} MiB ({total_mib / 1024:.3f} GiB) "
        f"across {len(held)} x {CHUNK_MIB} MiB blocks "
        f"(true budget is in [{total_mib}, {total_mib + CHUNK_MIB}) MiB)"
    )
    for handle in held:
        pinner.release(handle)
    return total_mib > 0


def main(argv=None):
    args = parse_args(argv)
    report_environment()
    lib = load_cudart(args.libcudart)
    pinner = Pinner(lib, args.mode)
    print(f"pin mode:               {args.mode}")

    if args.budget:
        return 0 if run_budget(pinner) else 1
    if args.probe:
        return 0 if run_probe(pinner, args.size_gb) else 1
    return 0 if run_single(pinner, args.size_gb, args.repeat, args.cycles) else 1


if __name__ == "__main__":
    sys.exit(main())

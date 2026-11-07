"""Hard MemoryMax budget for EasyICU bench scopes (peer rule, 2026-11-06 ~06:50Z).

A job is admitted only when the MemoryMax of every running bench scope plus its
own MemoryMax stays <= 260 GiB (executor <= 120 GiB, SICdb session <= 32 GiB,
~40 GiB reserve are budgeted elsewhere). Every scope has a finite MemoryMax.
Usage:
  memgate.py wait MAX_BYTES MODE   block until admitted (bench3.sh holds memgate.lock around it)
  memgate.py self MODE             inside a bench scope: own memory.max, own scope excluded
"""
import fcntl, glob, os, sys, time

BUDGET = 380 * 2**30  # temporary 380 until the seal (supervisor 2026-11-06 ~17:20Z; executor gate 70); restore 260 after
LOCK = "/home/zhuhb/easyicu_dev9/demo_sources/bench_local/memgate.lock"
SCOPES = f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/run-*.scope"


def _is_bench(pid):
    try:
        return b"bench_local.py" in open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return False


def committed(exclude=None):
    """Sum of memory.max over scopes running bench_local.py ('max' counts as the whole budget)."""
    total = 0
    for s in glob.glob(SCOPES):
        if s == exclude:
            continue
        try:
            mx = open(f"{s}/memory.max").read().strip()
            pids = open(f"{s}/cgroup.procs").read().split()
        except OSError:
            continue
        if any(_is_bench(p) for p in pids):
            total += BUDGET if mx == "max" else int(mx)
    return total


def wait(need, mode, exclude=None):
    if need <= 0:
        sys.exit("memgate: refusing a job without a finite MemoryMax")
    while True:
        used = committed(exclude)
        ok = used + need <= BUDGET
        stamp = time.strftime("%H:%M:%SZ", time.gmtime())
        print(f"{stamp} memgate {'admit' if ok else 'wait'} need={need / 2**30:.0f}G committed={used / 2**30:.0f}G budget={BUDGET // 2**30}G mode={mode}",
              file=sys.stderr, flush=True)
        if ok:
            return
        time.sleep(60)


def own_scope():
    return "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().split("::", 1)[1]


if __name__ == "__main__":
    if sys.argv[1] == "wait":
        wait(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "check":  # one non-blocking admission test (bench5.sh): exit 0 admit, 1 wait
        need = int(sys.argv[2])
        if need <= 0:
            sys.exit("memgate: refusing a job without a finite MemoryMax")
        used = committed()
        ok = used + need <= BUDGET
        stamp = time.strftime("%H:%M:%SZ", time.gmtime())
        print(f"{stamp} memgate {'admit' if ok else 'wait'} need={need / 2**30:.0f}G committed={used / 2**30:.0f}G budget={BUDGET // 2**30}G mode={sys.argv[3]}",
              file=sys.stderr, flush=True)
        sys.exit(0 if ok else 1)
    elif sys.argv[1] == "self":
        if os.environ.get("BENCH_GATED"):
            sys.exit(0)  # bench3.sh admitted this job before creating its scope
        scope = own_scope()
        mx = open(f"{scope}/memory.max").read().strip()
        with open(LOCK, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            wait(0 if mx == "max" else int(mx), sys.argv[2], exclude=scope)

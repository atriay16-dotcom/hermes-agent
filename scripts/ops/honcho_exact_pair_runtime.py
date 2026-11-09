"""OS and native boundaries of the one-time Honcho exact-pair executor, and its owner-side child.

Parent side (stdlib only, never imports Hermes code): /proc process identities, pidfds, mapped files,
loopback listener owners, loopback HTTP, ``systemctl`` and ``cgroup.procs`` reads.

Child side: ``python -I honcho_exact_pair_runtime.py child <op>`` runs ONE native operation in a fresh
interpreter as the installation owner, importing the checkout's own native modules (UpdateLock,
gateway control client, PM, cron), so every code version is used by its own subprocess. The payload
arrives in ``HX_PAYLOAD`` (paths and hashes only, never credentials); the answer is the last
``HXRESULT <json>`` line on stdout.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import select
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

RESULT = "HXRESULT "
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"  # hermes_cli/update_lock.py: adopt the held claim
_LOOPBACK = {"0100007F", "00000000000000000000000001000000"}
_HZ = os.sysconf("SC_CLK_TCK")
_SYS_PATH = "/usr/local/bin:/usr/bin:/bin"


# ---- process identity ---------------------------------------------------------------------------

def _stat_fields(pid: int):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return raw.rsplit(")", 1)[1].split()


def start_ticks(pid: int) -> int | None:
    """Kernel start time of a running (non-zombie) ``pid`` in clock ticks; None when gone."""
    fields = _stat_fields(pid)
    if fields is None or fields[0] in ("Z", "X"):
        return None
    return int(fields[19])


def _btime() -> int:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            return int(line.split()[1])
    raise RuntimeError("no btime in /proc/stat")


def proc_uid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def identity(pid: int) -> dict | None:
    """``{"pid", "ticks", "ct", "uid"}``; ``ct`` is spelled like ``update_pause_record.identity``."""
    ticks = start_ticks(pid)
    if ticks is None:
        return None
    return {"pid": pid, "ticks": ticks, "ct": f"ct:{_btime() + ticks / _HZ:.3f}", "uid": proc_uid(pid)}


def uptime_ticks() -> int:
    """Now, on the clock ``start_ticks`` uses (ticks since boot): fresh-generation ordering."""
    return int(float(Path("/proc/uptime").read_text().split()[0]) * _HZ)


def ct_seconds(ct: str) -> float:
    return float(str(ct).removeprefix("ct:"))


def same_generation(ident: dict | None) -> bool:
    return bool(ident) and start_ticks(ident["pid"]) == ident["ticks"]


def open_generation(ident: dict):
    """A pidfd pinned to exactly this generation, or None (gone / pid recycled)."""
    try:
        fd = os.pidfd_open(ident["pid"])
    except (ProcessLookupError, OSError):
        return None
    if not same_generation(ident):
        os.close(fd)
        return None
    return fd


def wait_exit(fd: int, timeout: float) -> bool:
    readable, _, _ = select.select([fd], [], [], max(0.0, timeout))
    return bool(readable)


def mapped_roots(pid: int, roots: list[str]) -> set[str] | None:
    """Which ``roots`` the process maps or executes from; None when its maps are unreadable."""
    try:
        text = Path(f"/proc/{pid}/maps").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return set()
    except OSError:
        return None
    paths = {line.split(None, 5)[5].removesuffix(" (deleted)") for line in text.splitlines() if len(line.split(None, 5)) == 6}
    try:
        paths.add(os.readlink(f"/proc/{pid}/exe"))
    except OSError:
        pass
    return {r for r in roots for p in paths if p.startswith(r.rstrip("/") + "/")}


def byte_holders(roots: list[str], owner_uid: int) -> tuple[dict[int, list[str]], list[int]]:
    """Every process mapping bytes under ``roots`` -> roots; plus owner/root-visible unreadable pids."""
    found, unreadable, me = {}, [], os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        pid = int(entry.name)
        hits = mapped_roots(pid, roots)
        if hits is None:
            if os.geteuid() == 0 or proc_uid(pid) == owner_uid:
                unreadable.append(pid)
        elif hits:
            found[pid] = sorted(hits)
    return found, unreadable


def listener_pids(port: int) -> set[int]:
    """PIDs holding a LISTEN socket on loopback ``port`` (wildcard binds do not count)."""
    inodes = set()
    for name in ("tcp", "tcp6"):
        try:
            lines = Path(f"/proc/net/{name}").read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            addr, hexport = f[1].split(":")
            if f[3] == "0A" and int(hexport, 16) == port and addr in _LOOPBACK:
                inodes.add(f[9])
    pids = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fds = os.listdir(entry / "fd")
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(entry / "fd" / fd)
            except OSError:
                continue
            if link.startswith("socket:[") and link[8:-1] in inodes:
                pids.add(int(entry.name))
    return pids


# ---- loopback HTTP / systemd ----------------------------------------------------------------------

def http_call(method: str, port: int, path: str, *, token: str | None = None, body: dict | None = None,
              timeout: float = 10.0) -> tuple[int | None, bytes]:
    """One loopback request. The session token travels only in its header; it is never returned."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    headers = {"X-Hermes-Session-Token": token} if token else {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    try:
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    except OSError:
        return None, b""
    finally:
        conn.close()


def as_json(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


_UNIT_PROPS = ("MainPID", "InvocationID", "ActiveState", "Restart", "KillMode", "ControlGroup")


def unit_show(systemctl: str, unit: str) -> dict | None:
    """Whitelisted unit properties only (never Environment or credentials)."""
    r = subprocess.run([systemctl, "show", unit, "--property=" + ",".join(_UNIT_PROPS)], capture_output=True,
                       check=False, text=True, timeout=60, env={"PATH": _SYS_PATH, "LANG": "C.UTF-8"})
    if r.returncode:
        return None
    props = {k: v for k, _, v in (ln.partition("=") for ln in r.stdout.splitlines()) if k in _UNIT_PROPS}
    props["MainPID"] = int(props.get("MainPID") or 0)
    return props


def unit_verb(systemctl: str, verb: str, unit: str) -> int:
    return subprocess.run([systemctl, verb, unit], capture_output=True, text=True, timeout=600, check=False,
                          env={"PATH": _SYS_PATH, "LANG": "C.UTF-8"}).returncode


def cgroup_procs(root: Path, cgroup: str) -> list[int] | None:
    try:
        return [int(x) for x in (Path(root) / cgroup.lstrip("/") / "cgroup.procs").read_text().split()]
    except (OSError, ValueError):
        return None


# ---- owner-side children --------------------------------------------------------------------------

class Owner:
    """Runs native operations as the installation owner, one fresh interpreter per operation."""

    def __init__(self, *, python: str, user: str, uid: int, gid: int, home: str, profile_home: str,
                 runuser: str = "/usr/sbin/runuser"):
        self.python, self.user, self.uid, self.gid = python, user, uid, gid
        self.home, self.profile_home, self.runuser = home, profile_home, runuser
        self.children: list[dict] = []

    def prefix(self) -> list[str]:
        if os.geteuid() == 0:
            return [self.runuser, "-u", self.user, "--"]
        if os.geteuid() != self.uid:
            raise PermissionError("owner_mismatch")
        return []

    def env(self, payload: dict | None = None, extra: dict | None = None) -> dict:
        env = {"PATH": _SYS_PATH, "LANG": "C.UTF-8", "HOME": self.home, "HERMES_HOME": self.profile_home,
               "PYTHONDONTWRITEBYTECODE": "1", "HX_PAYLOAD": json.dumps(payload or {})}
        if os.path.isdir(f"/run/user/{self.uid}"):
            env["XDG_RUNTIME_DIR"] = f"/run/user/{self.uid}"
        env.update(extra or {})
        return env

    def git(self, repo: str, *args: str) -> tuple[int, str]:
        r = subprocess.run(self.prefix() + ["git", "-C", repo, *args], capture_output=True, text=True,
                           timeout=600, check=False, env=self.env(), cwd="/")
        return r.returncode, r.stdout.strip()

    def run(self, op: str, payload: dict, *, cwd: str, extra: dict | None = None, timeout: float = 1800) -> dict:
        argv = self.prefix() + [self.python, "-I", str(Path(__file__).resolve()), "child", op]
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd,
                                env=self.env(payload, extra))
        out, err = proc.communicate(timeout=timeout)
        answers = [json.loads(ln[len(RESULT):]) for ln in out.splitlines() if ln.startswith(RESULT)]
        answer = answers[-1] if answers else None
        record = {"op": op, "pid": proc.pid, "exit": proc.returncode}
        self.children.append(record)
        if proc.returncode or answer is None or "error" in answer:
            detail = (answer or {}).get("error") or err.strip()[-800:]
            return {"ok": False, "error": detail, **record}
        return {"ok": True, "answer": answer, **record}

    def spawn_lock(self, checkout: str):
        argv = self.prefix() + [self.python, "-I", str(Path(__file__).resolve()), "child", "hold-lock"]
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, cwd=checkout, env=self.env({"checkout": checkout}))
        line = proc.stdout.readline()
        self.children.append({"op": "hold-lock", "pid": proc.pid, "exit": None})
        return proc, (json.loads(line[len(RESULT):]) if line.startswith(RESULT) else None)


def release_lock(proc) -> bool:
    if proc is None or proc.poll() is not None:
        return False
    proc.stdin.write("release\n")
    proc.stdin.close()
    out = proc.stdout.read()
    proc.wait(timeout=60)
    return '"released": true' in out


# ---- child operations (run as the owner, in a fresh interpreter) ---------------------------------

def _emit(obj: dict) -> None:
    print(RESULT + json.dumps(obj), flush=True)


def _native(checkout: str) -> None:
    sys.path.insert(0, checkout)


def _op_hold_lock(p: dict) -> None:
    _native(p["checkout"])
    from hermes_cli.update_lock import UpdateLock

    lock = UpdateLock(install_root=Path(p["checkout"]))
    ok = lock.acquire()
    _emit({"acquired": bool(ok), "pid": os.getpid(), "holder": None if ok else str(lock.holder)})
    if not ok:
        return
    if sys.stdin.readline().strip() == "release":
        lock.release()
        _emit({"released": True})
        return
    # Parent vanished: exit without releasing so the native stale-claim rules, not this child, decide.
    os._exit(3)


def _op_gateway(p: dict) -> None:
    _native(p["checkout"])
    from gateway.control_socket import query_gateway_control

    _emit({"answer": query_gateway_control(Path(os.environ["HERMES_HOME"]), p["verb"], params=p.get("params"),
                                           timeout=float(p.get("timeout", 10.0)))})


def _git(repo: str, *args: str, binary: bool = False):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=not binary, timeout=600, check=False)


def _op_git_switch(p: dict) -> None:
    co, sha = p["checkout"], p["sha"]
    if _git(co, "rev-parse", "--verify", "--quiet", sha + "^{commit}").returncode:
        raise RuntimeError(f"commit {sha} absent from checkout")
    if _git(co, "rev-parse", sha + "^{tree}").stdout.strip() != p["tree"]:
        raise RuntimeError("tree mismatch")
    r = _git(co, "checkout", "--quiet", "--detach", sha)
    if r.returncode:
        raise RuntimeError("checkout failed: " + r.stderr.strip()[-400:])
    _emit({"head": _git(co, "rev-parse", "HEAD").stdout.strip()})


def _op_git_fetch(p: dict) -> None:
    co, sha = p["checkout"], p["sha"]
    if _git(co, "rev-parse", "--verify", "--quiet", sha + "^{commit}").returncode:
        r = _git(co, "fetch", "--no-tags", "--quiet", p["source_repo"], sha)
        if r.returncode:
            raise RuntimeError("fetch failed: " + r.stderr.strip()[-400:])
    _emit({"tree": _git(co, "rev-parse", sha + "^{tree}").stdout.strip()})


def _fsync_write(path: Path, data: bytes, *, in_place: bool) -> None:
    if in_place:  # same inode, owner and mode (copy2 semantics of the retained rehearsal)
        with open(path, "r+b") as fh:
            fh.write(data)
            fh.truncate()
            fh.flush()
            os.fsync(fh.fileno())
        return
    tmp = path.with_name(path.name + ".hx-tmp")
    with open(tmp, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _append(path: Path, obj: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _op_plugin_apply(p: dict) -> None:
    live, backup = Path(p["live_dir"]), Path(p["backup"])
    base = backup / "baseline"
    base.mkdir(parents=True)
    meta = {}
    for f in p["files"]:
        dst = live / f["path"]
        if dst.is_file():
            st = dst.stat()
            (base / f["path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, base / f["path"])
            meta[f["path"]] = {"mode": stat.S_IMODE(st.st_mode), "atime_ns": st.st_atime_ns,
                               "mtime_ns": st.st_mtime_ns, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
    (backup / "baseline.json").write_text(json.dumps(meta))
    changed, created = [], []
    for f in p["files"]:
        data = _git(p["source_repo"], "cat-file", "blob", f"{p['commit']}:{f['path']}", binary=True).stdout
        if hashlib.sha256(data).hexdigest() != f["sha256"]:
            raise RuntimeError(f"frozen plugin blob mismatch: {f['path']}")
        dst = live / f["path"]
        if dst.is_file() and dst.read_bytes() == data:
            continue
        kind = "changed" if dst.is_file() else "created"
        _append(backup / "applied.ndjson", {"path": f["path"], "kind": kind})
        if kind == "created":
            dst.parent.mkdir(parents=True, exist_ok=True)
        _fsync_write(dst, data, in_place=kind == "changed")
        (changed if kind == "changed" else created).append(f["path"])
    _emit({"changed": changed, "created": created})


def _op_plugin_restore(p: dict) -> None:
    live, backup = Path(p["live_dir"]), Path(p["backup"])
    meta = json.loads((backup / "baseline.json").read_text())
    applied = [json.loads(x) for x in (backup / "applied.ndjson").read_text().splitlines()] \
        if (backup / "applied.ndjson").exists() else []
    archived = []
    for entry in applied:
        src = live / entry["path"]
        if entry["kind"] == "created" and src.exists():
            dest = backup / "candidate-only" / entry["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dest)
            archived.append(entry["path"])
    for rel, m in meta.items():
        dst, data = live / rel, (backup / "baseline" / rel).read_bytes()
        if dst.is_file() and dst.read_bytes() == data:
            continue
        _fsync_write(dst, data, in_place=dst.is_file())
        os.chmod(dst, m["mode"])
        os.utime(dst, ns=(m["atime_ns"], m["mtime_ns"]))
    bad = [rel for rel, m in meta.items() if hashlib.sha256((live / rel).read_bytes()).hexdigest() != m["sha256"]]
    if bad:
        raise RuntimeError(f"baseline plugin bytes not restored: {bad}")
    _emit({"restored": sorted(meta), "archived": archived})


def dist_version(venv: Path, normalized: str) -> str | None:
    hits = sorted(venv.glob(f"lib/python3*/site-packages/{normalized}-*.dist-info"))
    versions = {h.name[len(normalized) + 1:-len(".dist-info")] for h in hits}
    return versions.pop() if len(versions) == 1 else None


def _op_pm_selected(p: dict) -> None:
    _native(p["checkout"])
    from pm.environments import selected_venv

    _emit({"selected": str(selected_venv(Path(p["checkout"])))})


def _op_pm_sync(p: dict) -> None:
    _native(p["checkout"])
    import pm
    from pm.client import ensure_tools_for_sync
    from pm.environments import selected_venv
    from pm.plugin_inputs import Members

    members = Members({Path(p["plugin_dir"]): Path(p["plugin_dir"])})
    ensure_tools_for_sync()
    pm.sync_venv(["all"], explicit=True, plugins=members, evict_incompatible_plugins=False)
    if not pm.venv_is_current(plugins=members):
        raise RuntimeError("PM reports the environment is not current after sync")
    venv = selected_venv(Path(p["checkout"]))
    _emit({"selected": str(venv), "honcho_sdk": dist_version(venv, "honcho_ai")})


def _op_cron_read(p: dict) -> None:
    _native(p["checkout"])
    from cron.jobs import resolve_job_ref

    job = resolve_job_ref(p["job_id"])
    if job is None:
        _emit({"job": None})
        return
    repeat, schedule = job.get("repeat") or {}, job.get("schedule") or {}
    _emit({"job": {"id": job.get("id"), "enabled": job.get("enabled"), "completed": repeat.get("completed"),
                   "times": repeat.get("times"), "kind": schedule.get("kind"), "next_run_at": job.get("next_run_at"),
                   "run_claim": job.get("run_claim") is not None, "fire_claim": job.get("fire_claim") is not None}})


_OPS = {"hold-lock": _op_hold_lock, "gateway": _op_gateway, "git-switch": _op_git_switch,
        "git-fetch": _op_git_fetch, "plugin-apply": _op_plugin_apply, "plugin-restore": _op_plugin_restore,
        "pm-selected": _op_pm_selected, "pm-sync": _op_pm_sync, "cron-read": _op_cron_read}


def child_main(op: str) -> int:
    payload = json.loads(os.environ.get("HX_PAYLOAD") or "{}")
    try:
        _OPS[op](payload)
    except Exception as exc:  # the parent records the failure; never a partial success
        _emit({"error": f"{type(exc).__name__}: {exc}"[:800]})
        return 1
    return 0


def monotonic() -> float:
    return time.monotonic()


if __name__ == "__main__" and sys.argv[1:2] == ["child"]:
    sys.exit(child_main(sys.argv[2]))

"""One-time Honcho exact-pair cutover executor (operational artifact; never the installed payload).

Installs ONLY the frozen pair (core a61e8dc5, plugin 06664b49) into an existing checkout and plugin
directory, reusing native facilities: the Desktop serve handshake and RetirementFence permits, the
gateway control socket (identify / active-work / pause-for-update), UpdateLock, PM sync with explicit
Members and no eviction, the services' own unchanged units, and the paused one-shot post-start job.

``plan`` is read-only. ``execute --apply`` must run as root inside a transient
``hermes-honcho-cutover-*.service`` (never a hosting unit's cgroup) and claims a one-use operation
nonce. Every refusal, abort, rollback and hold writes a durable final result; only a fully verified
target sets ``acceptance_passed``. See HONCHO_EXACT_PAIR.md.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import signal
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

_RT_FILE = Path(__file__).resolve().with_name("honcho_exact_pair_runtime.py")


def _load_runtime():
    name = "honcho_exact_pair_runtime"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _RT_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rt = _load_runtime()

SCHEMA, RESULT_SCHEMA = "honcho-exact-pair/1", "honcho-exact-pair-result/1"
HOSTING_UNITS = ("hermes-gateway.service", "hermes-webui.service")
_TOKEN_RE = re.compile(r'window\.__HERMES_SESSION_TOKEN__\s*=\s*"([A-Za-z0-9_\-\.~]+)"')
_SHA_RE, _CT_RE = re.compile(r"[0-9a-f]{40}"), re.compile(r"ct:\d+\.\d{3}")
_KEYS = {"schema", "nonce", "deadline_utc", "artifact", "owner", "profile_home", "python", "core", "plugin",
         "services", "holders", "checkpoint", "handoff_dir", "final_result", "post_start"}
_SUBKEYS = {"artifact": {"executor_sha256", "runtime_sha256"}, "owner": {"user", "uid", "gid", "home"},
            "core": {"checkout", "source_repo", "commit", "tree", "baseline"},
            "plugin": {"live_dir", "source_repo", "commit", "tree", "files"},
            "services": {"gateway", "webui", "webui_health_url"}, "post_start": {"job_id", "hermes_bin"}}


@dataclass(frozen=True)
class Pins:
    core_commit: str
    core_tree: str
    plugin_commit: str
    plugin_tree: str
    plugin_files: int
    job_id: str
    honcho_sdk: str
    quiet_interval: float = 5.0
    quiet_wait: float = 300.0
    desktop_exit_budget: float = 15.0
    gateway_exit_budget: float = 180.0
    start_budget: float = 120.0
    gateway_unit: str = "hermes-gateway.service"
    webui_unit: str = "hermes-webui.service"
    permit_window: float = 25.0
    checkpoint_max_age: float = 900.0


LIVE_PINS = Pins("a61e8dc5aa33818d2995db58e8aea76fe05b174f", "242a5bb62684000d596b212ef4916cc6cb657b27",
                 "06664b491ab9a18edcf1dd48f68cf66e149483f8", "8d39010ccfecbfd6e8973fbf669ff1868ec7cc73", 64,
                 "865e1487a1ad", "2.8.0")


@dataclass(frozen=True)
class Host:
    systemctl: str = "/usr/bin/systemctl"
    cgroup_root: Path = Path("/sys/fs/cgroup")
    runuser: str = "/usr/sbin/runuser"


class ManifestError(Exception):
    def __init__(self, code: str, detail=None):
        super().__init__(code)
        self.code, self.detail = code, detail


class Stop(Exception):
    def __init__(self, code: str, detail=None):
        super().__init__(code)
        self.code, self.detail = code, detail


def _need(ok, code, detail=None):
    if not ok:
        raise ManifestError(code, detail)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _abs(value) -> bool:
    return isinstance(value, str) and value.startswith("/") and os.path.normpath(value) == value


def _owner(m: dict, host: Host):
    o = m["owner"]
    return rt.Owner(python=m["python"], user=o["user"], uid=o["uid"], gid=o["gid"], home=o["home"],
                    profile_home=m["profile_home"], runuser=host.runuser)


def load_manifest(path, pins: Pins, host: Host | None = None) -> dict:
    """Parse and bind every input; anything unknown, missing, weakened or unpinned fails closed."""
    host = host or Host()
    m = json.loads(Path(path).read_text())
    _need(isinstance(m, dict) and set(m) == _KEYS, "unknown_key", sorted(set(m) ^ _KEYS) if isinstance(m, dict) else None)
    for key, want in _SUBKEYS.items():
        _need(isinstance(m[key], dict) and set(m[key]) == want, "unknown_key", key)
    _need(m["schema"] == SCHEMA, "schema")
    _need(isinstance(m["nonce"], str) and re.fullmatch(r"[0-9a-f]{32,64}", m["nonce"]), "nonce")
    _need(m["artifact"]["executor_sha256"] == _sha(Path(__file__)) and m["artifact"]["runtime_sha256"] == _sha(_RT_FILE),
          "artifact_hash")
    core, plugin, svc, own = m["core"], m["plugin"], m["services"], m["owner"]
    _need((core["commit"], core["tree"], plugin["commit"], plugin["tree"], m["post_start"]["job_id"]) ==
          (pins.core_commit, pins.core_tree, pins.plugin_commit, pins.plugin_tree, pins.job_id), "pin_mismatch")
    _need(isinstance(core["baseline"], str) and _SHA_RE.fullmatch(core["baseline"]) and core["baseline"] != core["commit"],
          "baseline_schema")
    _need(all(_abs(v) for v in (m["profile_home"], m["python"], m["checkpoint"], m["handoff_dir"], own["home"],
                                core["checkout"], core["source_repo"], plugin["live_dir"], plugin["source_repo"],
                                m["post_start"]["hermes_bin"])), "path_schema")
    name = m["final_result"]
    _need(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", name), "path_schema")
    _need(isinstance(own["uid"], int) and isinstance(own["gid"], int) and isinstance(own["user"], str), "owner_schema")
    url = urllib.parse.urlsplit(str(svc["webui_health_url"]))
    _need(svc["gateway"] == pins.gateway_unit and svc["webui"] == pins.webui_unit and url.scheme == "http"
          and url.hostname == "127.0.0.1" and url.port and url.path == "/health", "service_schema")
    holders = m["holders"]
    _need(isinstance(holders, list) and all(
        isinstance(h, dict) and set(h) == {"pid", "ct", "port", "role"} and isinstance(h["pid"], int) and h["pid"] > 1
        and isinstance(h["ct"], str) and _CT_RE.fullmatch(h["ct"]) and isinstance(h["port"], int)
        and h["role"] in ("serve", "desktop-serve") for h in holders)
        and len({h["pid"] for h in holders}) == len(holders), "holder_schema")
    _need(_deadline(m) is not None, "schema", "deadline_utc")
    files = plugin["files"]
    paths = [f.get("path") for f in files] if isinstance(files, list) else []
    _need(len(paths) == pins.plugin_files and len(set(paths)) == len(paths) and all(
        isinstance(f, dict) and set(f) == {"path", "sha256"} and isinstance(f["path"], str)
        and not f["path"].startswith("/") and ".." not in f["path"].split("/") for f in files), "payload_manifest")
    owner = _owner(m, host)
    try:
        rc, listed = owner.git(plugin["source_repo"], "ls-tree", "-r", "--name-only", plugin["commit"])
        _need(rc == 0 and sorted(listed.splitlines()) == sorted(paths), "payload_manifest", "file list")
        rc, tree = owner.git(plugin["source_repo"], "rev-parse", plugin["commit"] + "^{tree}")
        _need(rc == 0 and tree == pins.plugin_tree, "payload_mismatch", "plugin tree")
        rc, tree = owner.git(core["source_repo"], "rev-parse", core["commit"] + "^{tree}")
        _need(rc == 0 and tree == pins.core_tree, "payload_mismatch", "core tree")
        for f in files:
            proc = rt.subprocess.run(owner.prefix() + ["git", "-C", plugin["source_repo"], "cat-file", "blob",
                                                       f"{plugin['commit']}:{f['path']}"], capture_output=True,
                                     timeout=120, env=owner.env(), cwd="/")
            _need(proc.returncode == 0 and hashlib.sha256(proc.stdout).hexdigest() == f["sha256"], "payload_manifest", f["path"])
    except PermissionError:
        raise ManifestError("owner_mismatch") from None
    return m


def _deadline(m):
    try:
        d = datetime.datetime.fromisoformat(str(m["deadline_utc"]))
    except ValueError:
        return None
    return d if d.tzinfo else None


def cgroup_guard(cgroup_text: str, environ, pins: Pins) -> None:
    """Refuse unless this is a transient cutover service outside every hosting unit's cgroup."""
    path = next((ln.split("::", 1)[1] for ln in cgroup_text.splitlines() if ln.startswith("0::")), "")
    segments = path.strip().split("/")
    if any(s in HOSTING_UNITS or s in (pins.gateway_unit, pins.webui_unit) for s in segments):
        raise ManifestError("hosting_cgroup", path)
    if not environ.get("INVOCATION_ID") or not re.fullmatch(r"hermes-honcho-cutover-[A-Za-z0-9_.-]+\.service", segments[-1]):
        raise ManifestError("not_transient_service", path)


def _checkpoint(m: dict, pins: Pins, err=ManifestError) -> dict:
    p = Path(m["checkpoint"])
    if p.is_symlink() or not p.is_file():
        raise err("checkpoint_missing")
    c = rt.as_json(p.read_bytes())
    if c.get("nonce") != m["nonce"] or c.get("controller_yielded") is not True:
        raise err("checkpoint_invalid")
    try:
        age = (_now() - datetime.datetime.fromisoformat(c["created_at"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        raise err("checkpoint_invalid") from None
    if not -60 <= age <= pins.checkpoint_max_age:
        raise err("checkpoint_stale", age)
    return {"age_seconds": round(age, 1)}


def atomic_write(path: Path, obj, owner=None) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode())
        if owner is not None and os.geteuid() == 0:
            os.fchown(fd, owner.uid, owner.gid)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class Journal:
    def __init__(self, path: Path):
        self.path = path

    def write(self, stage: str, **data) -> None:
        line = json.dumps({"at": _now().isoformat(), "stage": stage, **data}, sort_keys=True, default=str) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)


def _refused(code, detail=None) -> dict:
    return {"outcome": "refused", "reason": code, "detail": detail, "acceptance_passed": False, "holders": [], "children": []}


def execute(manifest_path, *, pins: Pins, host: Host, cgroup_text: str | None = None, environ=None, hooks=None) -> dict:
    environ = os.environ if environ is None else environ
    cgroup_text = Path("/proc/self/cgroup").read_text() if cgroup_text is None else cgroup_text
    try:
        m = load_manifest(manifest_path, pins, host)
        cgroup_guard(cgroup_text, environ, pins)
        if _now() > _deadline(m):
            raise ManifestError("deadline_passed")
        _checkpoint(m, pins)
        if (Path(m["handoff_dir"]) / m["final_result"]).exists():
            raise ManifestError("final_result_exists")
    except ManifestError as e:
        return _refused(e.code, e.detail)
    handoff = Path(m["handoff_dir"])
    try:
        fd = os.open(handoff / f"{m['nonce']}.operation.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return _refused("replay")
    os.write(fd, json.dumps({"nonce": m["nonce"], "claimed_at": _now().isoformat(), "pid": os.getpid(),
                             "invocation_id": environ.get("INVOCATION_ID"), "cgroup": cgroup_text.strip()}).encode())
    os.fsync(fd)
    os.close(fd)
    return Run(m, pins, host, hooks or {}).go()


class Run:
    def __init__(self, m: dict, pins: Pins, host: Host, hooks: dict):
        self.m, self.pins, self.host, self.hooks = m, pins, host, hooks
        self.owner = _owner(m, host)
        self.core, self.plugin = m["core"], m["plugin"]
        self.co = self.core["checkout"]
        self.handoff = Path(m["handoff_dir"])
        self.journal = Journal(self.handoff / f"{m['nonce']}.journal.ndjson")
        self.backup = self.handoff / f"{m['nonce']}.backup"
        self.phase, self.lock_proc, self.failure, self.recovery = "pre", None, None, None
        self.rows, self.svc, self.stopped, self.started = [], {}, [], []
        self.evidence = {"in_process_plugin_digest_api": "unavailable"}
        self.partial_retirement, self.env_restored, self.replacements = False, None, []
        self.started_at = _now().isoformat()

    # -- small native helpers
    def gateway(self, verb: str, timeout: float = 10.0):
        r = self.owner.run("gateway", {"checkout": self.co, "verb": verb, "timeout": timeout}, cwd=self.co)
        return r["answer"]["answer"] if r["ok"] else None

    def child(self, op: str, payload: dict, code: str, *, handoff: bool = False) -> dict:
        extra = {rt.HANDOFF_PID_ENV: str(self.lock_proc.pid)} if handoff and self.lock_proc else None
        r = self.owner.run(op, {"checkout": self.co, **payload}, cwd=self.co, extra=extra)
        self.journal.write("child", op=op, pid=r["pid"], exit=r["exit"], ok=r["ok"])
        if not r["ok"]:
            raise Stop(code, r["error"])
        return r["answer"]

    def show(self, unit: str) -> dict:
        props = rt.unit_show(self.host.systemctl, unit)
        if props is None:
            raise Stop("unit_unreadable", unit)
        return props

    def git(self, repo: str, *args: str) -> str:
        rc, out = self.owner.git(repo, *args)
        if rc:
            raise Stop("git_failed", " ".join(args))
        return out

    # -- preflight (read-only)
    def preflight(self) -> None:
        m, o = self.m, self.owner
        for p in (self.co, self.plugin["live_dir"], m["profile_home"], m["handoff_dir"]):
            if os.path.realpath(p) != p:
                raise Stop("symlink", p)
            if os.stat(p).st_uid != o.uid:
                raise Stop("owner_mismatch", p)
        live = Path(self.plugin["live_dir"])
        for f in self.plugin["files"]:
            parts = Path(f["path"]).parts
            if any((live.joinpath(*parts[:i + 1])).is_symlink() for i in range(len(parts))):
                raise Stop("symlink", f["path"])
        if self.git(self.co, "rev-parse", "HEAD") != self.core["baseline"]:
            raise Stop("baseline_mismatch")
        if self.git(self.co, "status", "--porcelain", "--untracked-files=no"):
            raise Stop("dirty_checkout")
        self.base_tree = self.git(self.co, "rev-parse", self.core["baseline"] + "^{tree}")
        self.old_venv = self.child("pm-selected", {}, "pm_unreadable")["selected"]
        for key in ("gateway", "webui"):
            props = self.show(m["services"][key])
            ident = rt.identity(props["MainPID"]) if props["MainPID"] else None
            if props.get("ActiveState") != "active" or ident is None:
                raise Stop("service_not_active", key)
            if ident["uid"] != o.uid:
                raise Stop("owner_mismatch", key)
            procs = rt.cgroup_procs(self.host.cgroup_root, props["ControlGroup"])
            if procs is None:
                raise Stop("cgroup_unreadable", key)
            self.svc[key] = {"unit": m["services"][key], "props": props, "ident": ident, "procs": procs}
        ans = self.gateway("identify")
        if not ans or ans.get("pid") != self.svc["gateway"]["ident"]["pid"]:
            raise Stop("gateway_identity_unproven")
        if ans.get("code_sha") != self.core["baseline"]:
            raise Stop("baseline_mismatch", "gateway boot identity")
        self.holder_idents = [self.verify_holder(h) for h in m["holders"]]
        found, unreadable = rt.byte_holders([self.co, self.old_venv], o.uid)
        allowed = set(self.svc["gateway"]["procs"]) | set(self.svc["webui"]["procs"]) | {h["pid"] for h in m["holders"]}
        if sorted(set(found) - allowed):
            raise Stop("undeclared_holder", sorted(set(found) - allowed))
        if unreadable and os.geteuid() == 0:
            raise Stop("holder_indeterminate", unreadable)
        self.job_ok(self.cron())
        self.journal.write("preflight", services={k: v["ident"] for k, v in self.svc.items()}, holders=self.holder_idents,
                           old_venv=self.old_venv, restart={k: v["props"].get("Restart") for k, v in self.svc.items()})

    def verify_holder(self, h: dict) -> dict:
        ident = rt.identity(h["pid"])
        if ident is None or abs(rt.ct_seconds(ident["ct"]) - rt.ct_seconds(h["ct"])) > 0.05 or ident["uid"] != self.owner.uid:
            raise Stop("holder_identity_changed", h["pid"])
        if rt.listener_pids(h["port"]) != {h["pid"]}:
            raise Stop("holder_listener_mismatch", h["pid"])
        return ident

    def cron(self) -> dict | None:
        return self.child("cron-read", {"job_id": self.m["post_start"]["job_id"]}, "cron_unreadable")["job"]

    @staticmethod
    def job_ok(job) -> None:
        if job is None:
            raise Stop("job_missing")
        if job["completed"] != 0:
            raise Stop("job_spent")
        if job["run_claim"] or job["fire_claim"]:
            raise Stop("job_claimed")
        if job["enabled"] is not False or job["next_run_at"] or job["kind"] != "once":
            raise Stop("job_not_paused")

    # -- lock and quiescence
    def lock(self) -> None:
        self.lock_proc, ans = self.owner.spawn_lock(self.co)
        if not ans or not ans.get("acquired"):
            raise Stop("lock_held", (ans or {}).get("holder"))
        self.journal.write("lock", pid=ans["pid"])
        tree = self.child("git-fetch", {"sha": self.core["commit"], "source_repo": self.core["source_repo"]},
                          "payload_fetch_failed", handoff=True)["tree"]
        if tree != self.pins.core_tree:
            raise Stop("payload_mismatch", "core tree in checkout")

    def quiesce(self) -> None:
        gw_pid, zeros = self.svc["gateway"]["ident"]["pid"], 0
        deadline = rt.monotonic() + self.pins.quiet_wait
        while True:
            ans = self.gateway("active-work")
            if not ans or not isinstance(ans.get("active_agents"), int):
                raise Stop("gateway_indeterminate")
            if ans.get("pid") != gw_pid or not rt.same_generation(self.svc["gateway"]["ident"]):
                raise Stop("gateway_generation_changed")
            zeros = zeros + 1 if ans["active_agents"] == 0 else 0
            if zeros >= 2:
                break
            if rt.monotonic() >= deadline:
                raise Stop("gateway_busy", ans["active_agents"])
            time.sleep(self.pins.quiet_interval)
        self.webui_quiet()
        _checkpoint(self.m, self.pins, Stop)
        self.journal.write("quiet")

    def webui_quiet(self) -> None:
        ui = self.svc["webui"]
        procs = rt.cgroup_procs(self.host.cgroup_root, ui["props"]["ControlGroup"])
        if procs is None or set(procs) != {ui["ident"]["pid"]} or not rt.same_generation(ui["ident"]):
            raise Stop("webui_nested_workers", procs)

    # -- Desktop holders: handshake, permits, graceful retirement
    def retire_desktop(self) -> None:
        sessions = []
        for h, ident in zip(self.m["holders"], self.holder_idents):
            row = {"kind": "desktop", "pre": {"pid": h["pid"], "ct": h["ct"], "port": h["port"], "role": h["role"]},
                   "action": "none", "exit_observed": False, "post": None,
                   "replacement_policy": "client_reconnect_only", "counted_as_updated": False}
            self.rows.append(row)
            sessions.append({"h": h, "ident": ident, "row": row, "token": None, "permit": None})
        try:
            for s in sessions:
                self.handshake(s)
            for s in sessions:
                self.verify_holder(s["h"])
                st, raw = rt.http_call("POST", s["h"]["port"], "/api/health/retirement", token=s["token"], body={"action": "prepare"})
                ans = rt.as_json(raw)
                if st != 200 or ans.get("ok") is not True or not ans.get("token"):
                    s["row"].update(action="refused", idle=ans.get("idle"))
                    raise Stop("permit_refused", {"pid": s["h"]["pid"], "idle": ans.get("idle")})
                s["permit"], s["prepared_at"] = ans["token"], rt.monotonic()
                s["row"]["action"] = "prepared"
            for s in sessions:
                if rt.monotonic() - s["prepared_at"] > self.pins.permit_window:
                    raise Stop("permit_expired", s["h"]["pid"])
                self.verify_holder(s["h"])
                st, raw = rt.http_call("POST", s["h"]["port"], "/api/health/retirement", token=s["token"],
                                       body={"action": "commit", "token": s["permit"]})
                if st != 200 or rt.as_json(raw).get("ok") is not True:
                    s["row"]["action"] = "commit_refused"
                    raise Stop("permit_commit_failed", s["h"]["pid"])
                s["row"]["action"] = "committed"
        except Stop:
            for s in sessions:
                if s["row"]["action"] in ("prepared", "commit_refused"):
                    st, raw = rt.http_call("POST", s["h"]["port"], "/api/health/retirement", token=s["token"],
                                           body={"action": "cancel", "token": s["permit"]})
                    ok = st == 200 and rt.as_json(raw).get("ok") is True
                    s["row"]["action"] = "cancelled" if ok else ("expired" if s["row"]["action"] == "commit_refused" else "cancel_failed")
                elif s["row"]["action"] == "committed":
                    s["row"]["action"] = "committed_not_retired"
                    self.partial_retirement = True
                s["permit"] = s["token"] = None
            raise
        for s in sessions:
            s["token"] = s["permit"] = None
            fd = rt.open_generation(s["ident"])
            if fd is None:
                s["row"].update(action="exited_after_commit", exit_observed=True)
                continue
            try:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                s["row"]["action"] = "retired_sigterm_after_commit"
                s["row"]["exit_observed"] = rt.wait_exit(fd, self.pins.desktop_exit_budget)
            finally:
                os.close(fd)
            if not s["row"]["exit_observed"]:
                self.partial_retirement = True
                raise Stop("holder_not_exited", s["h"]["pid"])
        self.journal.write("holders_retired", rows=self.rows)

    def handshake(self, s: dict) -> None:
        h = s["h"]
        self.verify_holder(h)
        st, raw = rt.http_call("GET", h["port"], "/api/health")
        health = rt.as_json(raw)
        if st != 200:
            raise Stop("holder_health", h["pid"])
        if health.get("auth_required") is not False:
            raise Stop("holder_gated", h["pid"])
        if health.get("commit") != self.core["baseline"]:
            raise Stop("holder_commit_mismatch", h["pid"])
        st, raw = rt.http_call("GET", h["port"], "/")
        match = _TOKEN_RE.search(raw.decode("utf-8", "replace")) if st == 200 else None
        if not match:
            raise Stop("credential_withheld", h["pid"])
        s["token"] = match.group(1)
        st, raw = rt.http_call("GET", h["port"], "/api/host/identity", token=s["token"])
        ident = rt.as_json(raw)
        if st != 200:
            raise Stop("auth_failed", h["pid"])
        if ident.get("pid") != h["pid"] or ident.get("role") not in ("serve", "desktop-serve"):
            raise Stop("holder_identity_changed", h["pid"])
        self.verify_holder(h)

    # -- gateway / WebUI stop
    def stop_gateway(self) -> None:
        gw = self.svc["gateway"]
        ans = self.gateway("active-work")
        if not ans or ans.get("active_agents") != 0 or ans.get("pid") != gw["ident"]["pid"]:
            raise Stop("gateway_busy" if ans else "gateway_indeterminate")
        fd = rt.open_generation(gw["ident"])
        if fd is None:
            raise Stop("gateway_generation_changed")
        try:
            ack = self.gateway("pause-for-update")
            if not ack or ack.get("pausing") is not True or ack.get("pid") != gw["ident"]["pid"]:
                raise Stop("gateway_pause_refused", ack)
            budget = min(self.pins.gateway_exit_budget, float(ack.get("drain_timeout") or 0) + 60.0)
            if not rt.wait_exit(fd, budget):
                raise Stop("gateway_drain_not_exited", budget)
        finally:
            os.close(fd)
        props = self.show(gw["unit"])
        if props["MainPID"] not in (0, gw["ident"]["pid"]):
            raise Stop("gateway_respawned", props["MainPID"])
        if rt.unit_verb(self.host.systemctl, "stop", gw["unit"]):
            raise Stop("gateway_stop_failed")
        self.stopped.append("gateway")
        if self.show(gw["unit"])["MainPID"]:
            raise Stop("gateway_stop_failed", "MainPID after stop")
        self.journal.write("gateway_stopped")

    def stop_webui(self) -> None:
        _checkpoint(self.m, self.pins, Stop)
        self.webui_quiet()
        ui = self.svc["webui"]
        if rt.unit_verb(self.host.systemctl, "stop", ui["unit"]):
            raise Stop("webui_stop_failed")
        self.stopped.append("webui")
        if self.show(ui["unit"])["MainPID"] or rt.same_generation(ui["ident"]):
            raise Stop("webui_stop_failed", "old generation alive")
        if _now() > _deadline(self.m):
            raise Stop("deadline_passed")
        self.journal.write("webui_stopped")

    # -- switch (point of no return) and verification
    def disk(self) -> dict:
        live = Path(self.plugin["live_dir"])
        return {"head": self.git(self.co, "rev-parse", "HEAD"), "tree": self.git(self.co, "rev-parse", "HEAD^{tree}"),
                "tracked_clean": self.git(self.co, "status", "--porcelain", "--untracked-files=no") == "",
                "plugin": {f["path"]: _sha(live / f["path"]) if (live / f["path"]).is_file() else None
                           for f in self.plugin["files"]}}

    def disk_ok(self, d: dict) -> bool:
        return (d["head"], d["tree"], d["tracked_clean"]) == (self.pins.core_commit, self.pins.core_tree, True) and all(
            d["plugin"][f["path"]] == f["sha256"] for f in self.plugin["files"])

    def switch(self) -> None:
        self.journal.write("switch_begin")
        self.child("git-switch", {"sha": self.core["commit"], "tree": self.pins.core_tree}, "source_switch_failed", handoff=True)
        self.child("plugin-apply", {"live_dir": self.plugin["live_dir"], "backup": str(self.backup),
                                    "source_repo": self.plugin["source_repo"], "commit": self.plugin["commit"],
                                    "files": self.plugin["files"]}, "plugin_install_failed", handoff=True)
        ans = self.child("pm-sync", {"plugin_dir": self.plugin["live_dir"]}, "pm_sync_failed", handoff=True)
        self.new_venv = ans["selected"]
        self.evidence.update(selected_environment={"before": self.old_venv, "after": self.new_venv},
                             honcho_sdk=ans["honcho_sdk"])
        if self.new_venv == self.old_venv:
            raise Stop("environment_not_reselected")
        if ans["honcho_sdk"] != self.pins.honcho_sdk:
            raise Stop("provider_sdk_mismatch", ans["honcho_sdk"])
        self.disk_pre = self.disk()
        if not self.disk_ok(self.disk_pre):
            raise Stop("payload_mismatch", "installed bytes")
        self.switch_done = rt.uptime_ticks()  # fresh processes must start after this tick
        self.journal.write("switch_done", disk=self.disk_pre, venv=self.new_venv)

    def wait_fresh(self, key: str, *, want_sha: str | None, after: int) -> dict | None:
        old, unit = self.svc[key]["ident"], self.svc[key]["unit"]
        deadline = rt.monotonic() + self.pins.start_budget
        while rt.monotonic() < deadline:
            props = rt.unit_show(self.host.systemctl, unit) or {"MainPID": 0}
            ident = rt.identity(props["MainPID"]) if props["MainPID"] else None
            if ident and ident["pid"] != old["pid"] and props.get("InvocationID") != self.svc[key]["props"]["InvocationID"] \
                    and ident["ticks"] >= after:
                if key == "gateway":
                    ans = self.gateway("identify")
                    if ans and ans.get("pid") == ident["pid"] and ans.get("state") == "running":
                        return {"ident": ident, "invocation": props["InvocationID"], "code_sha": ans.get("code_sha")}
                else:
                    url = urllib.parse.urlsplit(self.m["services"]["webui_health_url"])
                    st, _ = rt.http_call("GET", url.port, url.path, timeout=5.0)
                    if st == 200 and rt.same_generation(ident):
                        return {"ident": ident, "invocation": props["InvocationID"]}
            time.sleep(0.25)
        return None

    def start_gateway(self) -> None:
        unit = self.svc["gateway"]["unit"]
        self.started.append("gateway")
        if rt.unit_verb(self.host.systemctl, "start", unit):
            raise Stop("gateway_start_failed", "systemctl start")
        fresh = self.wait_fresh("gateway", want_sha=self.pins.core_commit, after=self.switch_done)
        if fresh is None:
            raise Stop("gateway_start_failed", "never identified a fresh generation")
        self.svc["gateway"]["post"] = fresh
        self.journal.write("gateway_started", fresh=fresh)

    def start_webui_and_verify(self) -> None:
        gw, ui = self.svc["gateway"], self.svc["webui"]
        self.started.append("webui")
        if rt.unit_verb(self.host.systemctl, "start", ui["unit"]):
            raise Stop("webui_start_failed", "systemctl start")
        fresh = self.wait_fresh("webui", want_sha=None, after=self.switch_done)
        if fresh is None:
            raise Stop("webui_start_failed", "no fresh healthy generation")
        ui["post"] = fresh
        if gw["post"]["code_sha"] != self.pins.core_commit:
            raise Stop("boot_identity_mismatch", gw["post"]["code_sha"])
        loaded = {}
        for key in ("gateway", "webui"):
            pid = self.svc[key]["post"]["ident"]["pid"]
            roots = rt.mapped_roots(pid, [self.new_venv, self.old_venv]) or set()
            loaded[key] = sorted(roots)
            if roots != {self.new_venv}:
                raise Stop("environment_not_loaded", {key: sorted(roots)})
        found, _ = rt.byte_holders([self.old_venv], self.owner.uid)
        if found:
            raise Stop("stale_holder", sorted(found))
        fresh_pids = {gw["post"]["ident"]["pid"], ui["post"]["ident"]["pid"]}
        for key in ("gateway", "webui"):
            fresh_pids |= set(rt.cgroup_procs(self.host.cgroup_root, self.show(self.svc[key]["unit"])["ControlGroup"]) or [])
        found, _ = rt.byte_holders([self.new_venv], self.owner.uid)
        self.replacements = sorted(set(found) - fresh_pids)
        disk_post = self.disk()
        if disk_post != self.disk_pre:
            raise Stop("payload_changed_during_boot")
        if self.lock_proc is None or self.lock_proc.poll() is not None:
            raise Stop("lock_lost_during_boot")
        if not all(rt.same_generation(self.svc[k]["post"]["ident"]) for k in ("gateway", "webui")):
            raise Stop("fresh_generation_changed")
        self.evidence.update(lock_held_across_boot=True, environment_mapped=loaded, disk_after_boot=disk_post,
                             gateway_boot_code_sha=gw["post"]["code_sha"],
                             provider_activation="selected environment mapped by both fresh processes; provider "
                                                 "distribution present in it; frozen plugin bytes unchanged across boot",
                             replacements_unverified=self.replacements)
        self.journal.write("verified", evidence=self.evidence)

    # -- recovery
    def restart_old(self, units) -> dict:
        rec = {"started": [], "verified": True}
        for key in units:
            unit = self.svc[key]["unit"]
            if rt.unit_verb(self.host.systemctl, "start", unit):
                rec["verified"] = False
                continue
            rec["started"].append(key)
            fresh = self.wait_fresh(key, want_sha=self.core["baseline"], after=0)
            if fresh is None or (key == "gateway" and fresh["code_sha"] != self.core["baseline"]):
                rec["verified"] = False
            rec[key] = fresh
        return rec

    def recover_services(self) -> None:
        self.recovery = {"kind": "restart_stopped_units_on_old_code", **self.restart_old([k for k in ("gateway", "webui") if k in self.stopped])}
        self.journal.write("recovery", **self.recovery)

    def rollback(self) -> bool:
        rec = {"kind": "payload_rollback", "steps": []}
        try:
            for key in reversed(self.started):
                rt.unit_verb(self.host.systemctl, "stop", self.svc[key]["unit"])
            self.child("git-switch", {"sha": self.core["baseline"], "tree": self.base_tree}, "rollback_source_failed", handoff=True)
            rec["steps"].append("baseline_checkout")
            if (self.backup / "baseline.json").exists():
                rec["plugin"] = self.child("plugin-restore", {"live_dir": self.plugin["live_dir"], "backup": str(self.backup)},
                                           "rollback_plugin_failed", handoff=True)
                rec["steps"].append("baseline_plugin_bytes")
            ans = self.child("pm-sync", {"plugin_dir": self.plugin["live_dir"]}, "rollback_pm_failed", handoff=True)
            rec["steps"].append("pm_resync")
            rec["selected_environment"] = ans["selected"]
            self.env_restored = False  # PM selects a NEW generation; the identical directory is never restored
            rec.update(self.restart_old(["gateway", "webui"]))
        except Stop as s:
            rec.update(verified=False, error=s.code, detail=s.detail)
        self.recovery = rec
        self.journal.write("rollback", **rec)
        return bool(rec.get("verified"))

    # -- orchestration
    def go(self) -> dict:
        try:
            try:
                self.preflight()
                self.lock()
                self.quiesce()
                self.phase = "holders"
                self.retire_desktop()
                self.phase = "services"
                self.stop_gateway()
                self.stop_webui()
                self.phase = "switch"
                self.switch()
                self.phase = "start_gateway"
                self.start_gateway()
                self.phase = "started"
                self.start_webui_and_verify()
                outcome, reason = "success", None
            except Exception as exc:  # every failure ends in a durable, truthful terminal outcome
                stop = exc if isinstance(exc, Stop) else Stop("internal_error", f"{type(exc).__name__}: {exc}")
                self.failure = {"reason": stop.code, "detail": stop.detail, "phase": self.phase}
                self.journal.write("failure", **self.failure)
                outcome, reason = self.resolve(stop)
        finally:
            released = rt.release_lock(self.lock_proc)
            self.journal.write("lock_released", released=released)
        return self.finish(outcome, reason)

    def resolve(self, stop: Stop) -> tuple[str, str]:
        if self.phase == "pre":
            return "refused", stop.code
        if self.phase in ("holders", "services"):
            self.recover_services()
            return "aborted_before_switch", stop.code
        if self.phase in ("switch", "start_gateway"):
            return ("rolled_back" if self.rollback() else "rollback_failed"), stop.code
        return "hold_after_start", stop.code

    def finish(self, outcome: str, reason) -> dict:
        success = outcome == "success"
        services = {}
        for key, s in self.svc.items():
            post = s.get("post")
            services[key] = {"pid": post["ident"]["pid"], "invocation": post["invocation"], "ct": post["ident"]["ct"]} if post else None
            self.rows.append({"kind": "service", "unit": s["unit"],
                              "pre": {"pid": s["ident"]["pid"], "ct": s["ident"]["ct"], "invocation": s["props"]["InvocationID"]},
                              "action": "stopped" if key in self.stopped else "untouched", "exit_observed": key in self.stopped,
                              "post": services[key], "replacement_policy": "unit_start", "counted_as_updated": success})
        retired = sum(1 for r in self.rows if r["kind"] == "desktop" and r["exit_observed"])
        recon = {"rows": self.rows, "summary": f"{2 if success else 0} services updated and verified, {retired} Desktop "
                                               f"generations retired, {len(self.replacements)} replacements observed"}
        atomic_write(self.handoff / f"{self.m['nonce']}.reconciliation.json", recon, self.owner)
        self.journal.write("reconciliation", summary=recon["summary"])
        record = {"schema": RESULT_SCHEMA, "nonce": self.m["nonce"], "outcome": outcome, "acceptance_passed": success,
                  "reason": reason, "failure": self.failure, "recovery": self.recovery,
                  "environment_directory_restored": self.env_restored, "partial_retirement": self.partial_retirement,
                  "release_gate": "open" if success else "held", "core": {"commit": self.pins.core_commit, "baseline": self.core["baseline"]},
                  "plugin": {"commit": self.pins.plugin_commit}, "artifact": self.m["artifact"], "services": services,
                  "holders": [r for r in self.rows if r["kind"] == "desktop"], "reconciliation": recon, "evidence": self.evidence,
                  "children": self.owner.children, "started_at": self.started_at, "ended_at": _now().isoformat()}
        if "before_final" in self.hooks:
            self.hooks["before_final"](record)
        atomic_write(self.handoff / self.m["final_result"], record, self.owner)
        self.journal.write("final", outcome=outcome, acceptance_passed=success)
        if success:
            record["post_start"] = post_start(None, pins=self.pins, host=self.host, _manifest=self.m)
        return record


def post_start(manifest_path, *, pins: Pins, host: Host, _manifest: dict | None = None) -> dict:
    """Resume the paused one-shot verifier once, only after a durable verified success and fresh services."""
    try:
        m = _manifest or load_manifest(manifest_path, pins, host)
    except ManifestError as e:
        return {"resumed": False, "reason": e.code}
    handoff, owner, co = Path(m["handoff_dir"]), _owner(m, host), m["core"]["checkout"]
    final = handoff / m["final_result"]
    if not (handoff / f"{m['nonce']}.operation.json").exists() or not final.exists():
        return {"resumed": False, "reason": "no_final_result"}
    rec = rt.as_json(final.read_bytes())
    if rec.get("nonce") != m["nonce"]:
        return {"resumed": False, "reason": "no_final_result"}
    if rec.get("outcome") != "success" or rec.get("acceptance_passed") is not True:
        return {"resumed": False, "reason": "not_success"}
    r = owner.run("cron-read", {"checkout": co, "job_id": m["post_start"]["job_id"]}, cwd=co)
    try:
        Run.job_ok(r["answer"]["job"] if r["ok"] else None)
    except Stop as s:
        return {"resumed": False, "reason": s.code}
    for key in ("gateway", "webui"):
        props = rt.unit_show(host.systemctl, m["services"][key]) or {}
        want = (rec.get("services") or {}).get(key) or {}
        ident = rt.identity(props.get("MainPID") or 0) if props.get("MainPID") else None
        if props.get("ActiveState") != "active" or not ident or ident["pid"] != want.get("pid") \
                or props.get("InvocationID") != want.get("invocation") or ident["ct"] != want.get("ct", ident["ct"]):
            return {"resumed": False, "reason": "services_not_fresh"}
    gw = owner.run("gateway", {"checkout": co, "verb": "identify"}, cwd=co)
    url = urllib.parse.urlsplit(m["services"]["webui_health_url"])
    if not gw["ok"] or (gw["answer"]["answer"] or {}).get("code_sha") != pins.core_commit \
            or rt.http_call("GET", url.port, url.path, timeout=5.0)[0] != 200:
        return {"resumed": False, "reason": "services_not_fresh"}
    claim = handoff / f"{m['nonce']}.post-start.json"
    try:
        os.close(os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
    except FileExistsError:
        return {"resumed": False, "reason": "post_start_spent"}
    cmd = owner.prefix() + [m["post_start"]["hermes_bin"], "cron", "resume", m["post_start"]["job_id"], "--run-now"]
    proc = rt.subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=owner.env(), cwd="/")
    back = owner.run("cron-read", {"checkout": co, "job_id": m["post_start"]["job_id"]}, cwd=co)
    record = {"command": cmd, "exit": proc.returncode, "stdout_tail": proc.stdout[-500:], "at": _now().isoformat(),
              "readback": back["answer"]["job"] if back["ok"] else None}
    atomic_write(claim, record, owner)
    return {"resumed": proc.returncode == 0, "reason": None if proc.returncode == 0 else "resume_failed", "record": record}


def plan(manifest_path, *, pins: Pins, host: Host) -> dict:
    """Read-only: bind the manifest and run the preflight checks; no lock, permit, stop or write."""
    try:
        m = load_manifest(manifest_path, pins, host)
    except ManifestError as e:
        return {"ok": False, "reason": e.code, "detail": e.detail}
    run = Run.__new__(Run)
    Run.__init__(run, m, pins, host, {})
    run.journal = type("NoJournal", (), {"write": staticmethod(lambda *a, **k: None)})()
    try:
        run.preflight()
    except Stop as s:
        return {"ok": False, "reason": s.code, "detail": s.detail}
    return {"ok": True, "services": {k: v["ident"] for k, v in run.svc.items()}, "holders": run.holder_idents,
            "old_venv": run.old_venv}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="honcho_exact_pair", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "execute", "post-start"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True, type=Path)
        if name == "execute":
            p.add_argument("--apply", action="store_true", help="perform the live cutover (root, transient unit only)")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        r = plan(a.manifest, pins=LIVE_PINS, host=Host())
        print(json.dumps(r, default=str))
        return 0 if r["ok"] else 2
    if a.cmd == "post-start":
        r = post_start(a.manifest, pins=LIVE_PINS, host=Host())
        print(json.dumps(r, default=str))
        return 0 if r["resumed"] else 2
    if not a.apply or os.geteuid() != 0:
        print(json.dumps(_refused("apply_flag_required" if not a.apply else "root_required")))
        return 2
    r = execute(a.manifest, pins=LIVE_PINS, host=Host())
    print(json.dumps({k: r.get(k) for k in ("outcome", "reason", "acceptance_passed", "release_gate")}))
    return 0 if r["outcome"] == "success" else (2 if r["outcome"] == "refused" else 1)


if __name__ == "__main__":
    sys.exit(main())

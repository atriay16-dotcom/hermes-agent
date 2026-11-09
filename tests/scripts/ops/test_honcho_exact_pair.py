"""Behaviour tests for the one-time Honcho exact-pair cutover executor (scripts/ops).

FIXTURE BOUNDARIES (labelled; none of this is live proof): the core/plugin repositories, the native
gateway control client, ``UpdateLock``, PM and cron modules inside the temporary core checkout, the
``systemctl`` shim with its ``cgroup.procs`` files, and the ``hermes`` CLI shim.
REAL here: git repositories and checkouts, OS child processes and their identities (/proc start
ticks, pidfds, mmaps, loopback listener ownership), loopback HTTP, and the shipped
``hermes_cli.backend_retirement.RetirementFence`` served by every fixture Desktop serve.
Nothing here reads or writes the installed checkout, services, credentials or cron registry.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import pwd
import signal
import socket
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

pytestmark = pytest.mark.platforms("linux")

REPO = Path(__file__).resolve().parents[3]
OPS = REPO / "scripts" / "ops"
PY = sys.executable


def _load(name):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hx = _load("honcho_exact_pair")

GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_NOSYSTEM": "1"}
# Fixture unit names: the conftest live-system guard blocks any real hermes-* unit verb.
GW, UI = "fixture-gateway.service", "fixture-webui.service"
CUTOVER_CGROUP = "0::/system.slice/hermes-honcho-cutover-20261109.service\n"


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env={**os.environ, **GIT_ENV}).stdout.strip()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def meta(p: Path) -> tuple:
    s = p.stat()
    return (s.st_uid, s.st_gid, s.st_mode, s.st_mtime_ns, s.st_ino, sha256(p))


def ct(pid: int) -> str:
    return f"ct:{psutil.Process(pid).create_time():.3f}"


def write_tree(root: Path, files: dict):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


# ---- fixture native modules inside the temporary core checkout -------------------------------

FIX_CONTROL = r'''
import json, os, select, signal, subprocess
from pathlib import Path
def _home(): return Path(os.environ["HERMES_HOME"])
def _cfg():
    p = _home() / "fixture-gateway.json"
    return json.loads(p.read_text()) if p.exists() else {}
def _alive(pid):
    try: s = Path(f"/proc/{pid}/stat").read_text()
    except OSError: return False
    return s.rsplit(")", 1)[1].split()[0] != "Z"
def query_gateway_control(home, verb, *, params=None, timeout=5.0):
    p = _home() / "fixture-gateway-boot.json"
    boot = json.loads(p.read_text()) if p.exists() else None
    if not boot or not _alive(boot["pid"]): return None
    cfg = _cfg()
    if verb == "identify":
        sha = cfg.get("identify_map", {}).get(boot["code_sha"], boot["code_sha"])
        return {"pid": boot["pid"], "code_sha": sha, "state": "running", "instance_id": boot["instance"]}
    if verb == "active-work":
        seq = cfg.get("active", [0]); n = cfg.get("_n", 0); cfg["_n"] = n + 1
        (_home() / "fixture-gateway.json").write_text(json.dumps(cfg))
        val = seq[min(n, len(seq) - 1)]
        return None if val is None else {"active_agents": val, "active_work": [{"kind": "chat"}] * val, "pid": boot["pid"]}
    if verb == "pause-for-update":
        mode = cfg.get("pause", "exit")
        if mode in ("exit", "respawn"):
            fd = os.pidfd_open(boot["pid"]); os.kill(boot["pid"], signal.SIGTERM); select.select([fd], [], [], 10); os.close(fd)
        if mode == "respawn":
            subprocess.run([cfg["systemctl"], "__respawn", "fixture-gateway.service"], check=True)
        return {"pausing": True, "already_stopping": False, "pid": boot["pid"], "drain_timeout": 0.5}
    return None
'''

FIX_LOCK = r'''
import fcntl, os
from pathlib import Path
class UpdateLock:
    def __init__(self, *, path=None, install_root=None, checkout_first=True):
        self.install_root = Path(install_root); self.holder = None; self._fd = None
    def acquire(self):
        fd = os.open(self.install_root / ".git" / "fixture-update.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd); self.holder = "held"; return False
        self._fd = fd
        (Path(os.environ["HERMES_HOME"]) / ".update_in_progress").write_text(str(os.getpid()))
        return True
    def release(self):
        (Path(os.environ["HERMES_HOME"]) / ".update_in_progress").unlink(missing_ok=True)
        fcntl.flock(self._fd, fcntl.LOCK_UN); os.close(self._fd)
'''

FIX_PM_ENV = r'''
import os
from pathlib import Path
def _state(): return Path(os.environ["HERMES_HOME"]) / "pm-state"
def selected_venv(project_root): return Path((_state() / "selected").read_text())
def make_gen(g):
    d = g / "lib" / "python3.12" / "site-packages" / "honcho_ai-2.8.0.dist-info"
    d.mkdir(parents=True); (d / "METADATA").write_text("Name: honcho-ai\nVersion: 2.8.0\n")
    (g / "lib" / "marker.so").write_bytes(b"\0" * 4096)
'''

FIX_PM = r'''
from pm.environments import _state, make_gen
def sync_venv(extras, *, explicit, plugins, evict_incompatible_plugins):
    assert explicit is True and evict_incompatible_plugins is False and plugins.mapping
    st = _state()
    if (st / "fail-next-sync").exists():
        (st / "fail-next-sync").unlink(); raise RuntimeError("fixture: resolver offline")
    g = st / f"gen-{len(list(st.glob('gen-*')))}"; make_gen(g); (st / "selected").write_text(str(g))
def venv_is_current(*, plugins): return True
'''

FIX_CRON = r'''
import json, os
from pathlib import Path
def resolve_job_ref(job_id):
    for j in json.loads((Path(os.environ["HERMES_HOME"]) / "cron" / "jobs.json").read_text())["jobs"]:
        if j["id"] == job_id: return j
    return None
'''

SHIM = r'''#!@PY@
import json, os, select, signal, subprocess, sys, uuid
from pathlib import Path
ST = Path(@UNITS@); CG = Path(@CG@); LEDGER = Path(@LEDGER@)
def load(): return json.loads(ST.read_text())
def save(d):
    t = ST.with_suffix(".tmp"); t.write_text(json.dumps(d)); os.replace(t, ST)
def alive(pid):
    if not pid: return False
    try: s = Path(f"/proc/{pid}/stat").read_text()
    except OSError: return False
    return s.rsplit(")", 1)[1].split()[0] != "Z"
def procs(u, d):
    p = CG / d[u]["cgroup"].lstrip("/") / "cgroup.procs"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{x}\n" for x in ([d[u]["pid"]] if alive(d[u]["pid"]) else []) + d[u].get("extra", [])))
cmd, unit = sys.argv[1], sys.argv[2]
d = load()
if cmd not in ("show", "__respawn"):
    with LEDGER.open("a") as f: f.write(json.dumps([cmd, unit]) + "\n")
if cmd == "show":
    u = d[unit]; pid = u["pid"] if alive(u["pid"]) else 0
    print(f"MainPID={pid}\nInvocationID={u['invocation'] if pid else ''}\nActiveState={'active' if pid else 'inactive'}\n"
          f"Restart=always\nKillMode={'mixed' if 'gateway' in unit else 'control-group'}\nControlGroup={u['cgroup']}")
elif cmd in ("start", "__respawn"):
    log = open(ST.parent / (unit + ".log"), "ab")
    p = subprocess.Popen(d[unit]["argv"], stdout=subprocess.PIPE, stderr=log, start_new_session=True)
    p.stdout.readline(); p.stdout.close()
    d[unit]["pid"] = p.pid; d[unit]["invocation"] = uuid.uuid4().hex; save(d); procs(unit, d)
elif cmd == "stop":
    pid = d[unit]["pid"]
    if alive(pid):
        fd = os.pidfd_open(pid); os.kill(pid, signal.SIGTERM); select.select([fd], [], [], 10); os.close(fd)
    d[unit]["pid"] = 0; save(d); procs(unit, d)
'''

SERVICE = r'''
import http.server, json, mmap, os, signal, subprocess, sys, threading, uuid
from pathlib import Path
role, home, checkout = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
cfgp = home / f"fixture-{role}.json"
cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True, text=True).stdout.strip()
if sha in cfg.get("crash_on", []): sys.exit(1)
sel = Path((home / "pm-state" / "selected").read_text())
fh = open(sel / "lib" / "marker.so", "rb"); held = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
signal.signal(signal.SIGTERM, lambda *a: os._exit(75 if role == "gateway" else 0))
if role == "webui":
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200 if self.path == "/health" else 404)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", cfg["port"]), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
boot = home / f"fixture-{role}-boot.json"; tmp = boot.with_suffix(".tmp")
tmp.write_text(json.dumps({"pid": os.getpid(), "code_sha": sha, "instance": uuid.uuid4().hex})); os.replace(tmp, boot)
print("READY", flush=True)
while True: signal.pause()
'''

SERVE = r'''
import http.server, importlib.util, json, mmap, os, sys, time, types
from pathlib import Path
worktree, home, name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
cfgp = home / f"fixture-serve-{name}.json"
def cfg(): return json.loads(cfgp.read_text())
pkg = types.ModuleType("hermes_cli"); pkg.__path__ = []
idle = types.ModuleType("hermes_cli.web_server_idle_proof")
idle.idle_proof = lambda: {"idle": cfg().get("idle", True), "reason": None}
sys.modules["hermes_cli"] = pkg; sys.modules["hermes_cli.web_server_idle_proof"] = idle
spec = importlib.util.spec_from_file_location("hermes_cli.backend_retirement", Path(worktree) / "hermes_cli" / "backend_retirement.py")
br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)
fence = br.RetirementFence(); skew = [0.0]; fence._now = lambda: time.monotonic() + skew[0]
TOKEN = os.urandom(16).hex()
sel = Path((home / "pm-state" / "selected").read_text())
fh = open(sel / "lib" / "marker.so", "rb"); held = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
log = home / f"fixture-serve-{name}.requests"; state = home / f"fixture-serve-{name}.fence"
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, obj=None, html=None):
        body = html.encode() if html is not None else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def authed(self):
        return not cfg().get("reject_auth") and self.headers.get("X-Hermes-Session-Token") == TOKEN
    def do_GET(self):
        with log.open("a") as f: f.write("GET " + self.path + "\n")
        c = cfg()
        if self.path == "/api/health":
            return self.reply(200, {"ok": True, "commit": c["commit"], "auth_required": bool(c.get("gated"))})
        if self.path == "/":
            tok = "" if c.get("gated") or c.get("withheld") else f'window.__HERMES_SESSION_TOKEN__="{TOKEN}";'
            return self.reply(200, html=f"<html><script>{tok}window.__HERMES_AUTH_REQUIRED__=false;</script></html>")
        if self.path == "/api/host/identity":
            if not self.authed(): return self.reply(401, {"detail": "Unauthorized"})
            return self.reply(200, {"ok": True, "protocolVersion": 1, "pid": os.getpid(), "role": c.get("role", "serve")})
        self.reply(404, {})
    def do_POST(self):
        with log.open("a") as f: f.write("POST " + self.path + "\n")
        if self.path != "/api/health/retirement" or not self.authed(): return self.reply(401, {"detail": "Unauthorized"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        c = cfg(); a = body["action"]
        if a == "prepare":
            r = fence.prepare()
            if c.get("expire_after_prepare"): skew[0] += 31.0
        elif a == "commit":
            r = fence.commit(body["token"] + ("-mangled" if c.get("mangle_commit") else ""))
        else:
            r = fence.cancel(body["token"])
        state.write_text(json.dumps({"committed": fence._committed, "permit_open": fence._token is not None and not fence._committed}))
        self.reply(200, r)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
print(srv.server_address[1], flush=True)
srv.serve_forever()
'''

HERMES_SHIM = r'''#!@PY@
import datetime, json, os, sys
from pathlib import Path
home = Path(os.environ["HERMES_HOME"])
with (home / "fixture-hermes-ledger").open("a") as f: f.write(json.dumps(sys.argv[1:]) + "\n")
p = home / "cron" / "jobs.json"; d = json.loads(p.read_text())
for j in d["jobs"]:
    if j["id"] == sys.argv[3]:
        j["enabled"] = True; j["next_run_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
p.write_text(json.dumps(d))
print("Resumed job")
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class World:
    """Temporary installation: core checkout, plugin dir, profile home, two units and Desktop serves."""

    def __init__(self, tmp: Path, serves: list[dict]):
        self.tmp, self.procs = tmp, []
        self.home = tmp / "profile"; self.home.mkdir()
        self.owner_home = tmp / "owner-home"; self.owner_home.mkdir()
        self.handoff = tmp / "handoff"; self.handoff.mkdir()
        # core source repo: baseline and target commits carrying the fixture native modules
        src = tmp / "core-src"; src.mkdir(); git(src, "init", "-q", "-b", "main")
        write_tree(src, {"gateway/__init__.py": "", "gateway/control_socket.py": FIX_CONTROL,
                         "hermes_cli/__init__.py": "", "hermes_cli/update_lock.py": FIX_LOCK,
                         "pm/__init__.py": FIX_PM, "pm/environments.py": FIX_PM_ENV,
                         "pm/client.py": "def ensure_tools_for_sync():\n    return None\n",
                         "pm/plugin_inputs.py": "class Members:\n    def __init__(self, m):\n        self.mapping = dict(m)\n",
                         "cron/__init__.py": "", "cron/jobs.py": FIX_CRON,
                         "keep.txt": "unchanged tracked\n", "changed.txt": "old\n"})
        git(src, "add", "-A"); git(src, "commit", "-q", "-m", "baseline")
        self.base = git(src, "rev-parse", "HEAD")
        write_tree(src, {"changed.txt": "new\n", "added.txt": "target only\n"})
        git(src, "add", "-A"); git(src, "commit", "-q", "-m", "target")
        self.target, self.target_tree = git(src, "rev-parse", "HEAD"), git(src, "rev-parse", "HEAD^{tree}")
        self.checkout = tmp / "hermes-agent"
        git(tmp, "clone", "-q", "--no-hardlinks", str(src), str(self.checkout))
        git(self.checkout, "checkout", "-q", "--detach", self.base)
        (self.checkout / ".claude").mkdir(mode=0o700)
        self.marker = self.checkout / ".claude" / "owner-marker"; self.marker.write_text("untracked owner work\n")
        self.core_src = src
        # plugin: candidate commit and an older installed copy plus untracked user data
        psrc = tmp / "plugin-src"; psrc.mkdir(); git(psrc, "init", "-q", "-b", "main")
        write_tree(psrc, {"__init__.py": "VERSION = 'new'\n", "pyproject.toml": "[project]\nname='honcho'\n",
                          "plugin.yaml": "version: 1.1.0\n", "provider.py": "candidate only\n"})
        git(psrc, "add", "-A"); git(psrc, "commit", "-q", "-m", "plugin")
        self.pcommit, self.ptree = git(psrc, "rev-parse", "HEAD"), git(psrc, "rev-parse", "HEAD^{tree}")
        self.plugin_src = psrc
        self.plugin = tmp / "plugins" / "honcho"
        write_tree(self.plugin, {"__init__.py": "VERSION = 'old'\n", "pyproject.toml": "[project]\nname='honcho'\n",
                                 "plugin.yaml": "version: 1.0.0\n", "user-notes.txt": "keep me\n"})
        self.pfiles = sorted(git(psrc, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
        # PM state, cron registry, shims
        st = self.home / "pm-state"; st.mkdir()
        spec = importlib.util.spec_from_file_location("fixture_pm_env", src / "pm" / "environments.py")
        pmenv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pmenv)
        pmenv.make_gen(st / "gen-0"); (st / "selected").write_text(str(st / "gen-0"))
        self.old_venv = st / "gen-0"
        (self.home / "cron").mkdir()
        self.set_job()
        self.hermes = tmp / "bin" / "hermes"; self.hermes.parent.mkdir()
        self.hermes.write_text(HERMES_SHIM.replace("@PY@", PY)); self.hermes.chmod(0o755)
        (tmp / "service.py").write_text(SERVICE); (tmp / "serve.py").write_text(SERVE)
        self.cg = tmp / "cgroup"; self.units = tmp / "units.json"; self.ledger = tmp / "systemctl-ledger"
        self.systemctl = tmp / "bin" / "systemctl"
        self.systemctl.write_text(SHIM.replace("@PY@", PY).replace("@UNITS@", repr(str(self.units)))
                                  .replace("@CG@", repr(str(self.cg))).replace("@LEDGER@", repr(str(self.ledger))))
        self.systemctl.chmod(0o755)
        self.port = _free_port()
        self.cfg("webui", {"port": self.port}); self.cfg("gateway", {"systemctl": str(self.systemctl)})
        argv = lambda role: [PY, str(tmp / "service.py"), role, str(self.home), str(self.checkout)]
        self.units.write_text(json.dumps({
            GW: {"argv": argv("gateway"), "pid": 0, "invocation": "", "cgroup": "/system.slice/" + GW},
            UI: {"argv": argv("webui"), "pid": 0, "invocation": "", "cgroup": "/system.slice/" + UI}}))
        for unit in (GW, UI):
            subprocess.run([str(self.systemctl), "start", unit], check=True)
        self.ledger.unlink()
        self.holders = []
        for i, scfg in enumerate(serves):
            name = f"s{i}"
            (self.home / f"fixture-serve-{name}.json").write_text(json.dumps({"commit": self.base, **scfg}))
            p = subprocess.Popen([PY, str(tmp / "serve.py"), str(REPO), str(self.home), name], stdout=subprocess.PIPE)
            self.procs.append(p)
            port = int(p.stdout.readline())
            self.holders.append({"pid": p.pid, "ct": ct(p.pid), "port": port, "role": scfg.get("role", "desktop-serve")})
        self.nonce = os.urandom(16).hex()
        self.checkpoint = self.handoff / "QUIET-CHECKPOINT.json"
        self.write_checkpoint()
        user = pwd.getpwuid(os.getuid())
        self.manifest = {
            "schema": "honcho-exact-pair/1", "nonce": self.nonce,
            "deadline_utc": (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)).isoformat(),
            "artifact": {"executor_sha256": sha256(OPS / "honcho_exact_pair.py"),
                         "runtime_sha256": sha256(OPS / "honcho_exact_pair_runtime.py")},
            "owner": {"user": user.pw_name, "uid": os.getuid(), "gid": os.getgid(), "home": str(self.owner_home)},
            "profile_home": str(self.home), "python": PY,
            "core": {"checkout": str(self.checkout), "source_repo": str(src), "commit": self.target,
                     "tree": self.target_tree, "baseline": self.base},
            "plugin": {"live_dir": str(self.plugin), "source_repo": str(psrc), "commit": self.pcommit, "tree": self.ptree,
                       "files": [{"path": f, "sha256": sha256(psrc / f)} for f in self.pfiles]},
            "services": {"gateway": GW, "webui": UI, "webui_health_url": f"http://127.0.0.1:{self.port}/health"},
            "holders": self.holders, "checkpoint": str(self.checkpoint), "handoff_dir": str(self.handoff),
            "final_result": "CUTOVER-RUNTIME-ACCEPTANCE.json",
            "post_start": {"job_id": "fixturejob01", "hermes_bin": str(self.hermes)}}
        self.pins = hx.Pins(core_commit=self.target, core_tree=self.target_tree, plugin_commit=self.pcommit,
                            plugin_tree=self.ptree, plugin_files=len(self.pfiles), job_id="fixturejob01",
                            honcho_sdk="2.8.0", quiet_interval=0.2, quiet_wait=0.8, desktop_exit_budget=5.0,
                            gateway_exit_budget=3.0, start_budget=4.0, gateway_unit=GW, webui_unit=UI)
        self.host = hx.Host(systemctl=str(self.systemctl), cgroup_root=self.cg)
        self.save()

    # -- helpers
    def cfg(self, role, data):
        (self.home / f"fixture-{role}.json").write_text(json.dumps(data))

    def serve_cfg(self, i, **kw):
        p = self.home / f"fixture-serve-s{i}.json"
        p.write_text(json.dumps({**json.loads(p.read_text()), **kw}))

    def set_job(self, **kw):
        job = {"id": "fixturejob01", "enabled": False, "next_run_at": None, "repeat": {"times": 1, "completed": 0},
               "schedule": {"kind": "once"}, "run_claim": None, "fire_claim": None, **kw}
        (self.home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [job]}))

    def write_checkpoint(self, age=0.0):
        at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=age)
        self.checkpoint.write_text(json.dumps({"nonce": self.nonce, "created_at": at.isoformat(), "controller_yielded": True}))

    def save(self):
        self.manifest_path = self.tmp / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))

    def run(self, **kw):
        return hx.execute(self.manifest_path, pins=self.pins, host=self.host, cgroup_text=CUTOVER_CGROUP,
                          environ={"INVOCATION_ID": "0123456789abcdef"}, **kw)

    def unit(self, u):
        return json.loads(self.units.read_text())[u]

    def boot(self, role):
        return json.loads((self.home / f"fixture-{role}-boot.json").read_text())

    def ledger_lines(self):
        return [json.loads(x) for x in self.ledger.read_text().splitlines()] if self.ledger.exists() else []

    def hermes_calls(self):
        p = self.home / "fixture-hermes-ledger"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def fence(self, i):
        p = self.home / f"fixture-serve-s{i}.fence"
        return json.loads(p.read_text()) if p.exists() else None

    def requests(self, i):
        p = self.home / f"fixture-serve-s{i}.requests"
        return p.read_text().splitlines() if p.exists() else []

    def final(self):
        p = self.handoff / "CUTOVER-RUNTIME-ACCEPTANCE.json"
        return json.loads(p.read_text()) if p.exists() else None

    def cleanup(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        for u in json.loads(self.units.read_text()).values():
            for pid in [u["pid"], *u.get("extra", [])]:
                if pid and psutil.pid_exists(pid):
                    # orphaned fixture services (recorded in this test's own units.json)
                    fd = os.pidfd_open(pid)
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                    os.close(fd)


@pytest.fixture
def world(tmp_path, request):
    w = World(tmp_path, getattr(request, "param", [{"role": "desktop-serve"}]))
    yield w
    w.cleanup()


def _serves(*cfgs):
    return pytest.mark.parametrize("world", [list(cfgs)], indirect=True)


# ---- exact target, manifest and launch guards --------------------------------------------------

def test_live_pins_are_the_frozen_pair_and_plan_rejects_any_other_target(world, capsys):
    assert (hx.LIVE_PINS.core_commit, hx.LIVE_PINS.core_tree) == (
        "a61e8dc5aa33818d2995db58e8aea76fe05b174f", "242a5bb62684000d596b212ef4916cc6cb657b27")
    assert (hx.LIVE_PINS.plugin_commit, hx.LIVE_PINS.plugin_tree, hx.LIVE_PINS.plugin_files) == (
        "06664b491ab9a18edcf1dd48f68cf66e149483f8", "8d39010ccfecbfd6e8973fbf669ff1868ec7cc73", 64)
    assert hx.LIVE_PINS.job_id == "865e1487a1ad"
    assert hx.main(["plan", "--manifest", str(world.manifest_path)]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "pin_mismatch"
    # execute without --apply never mutates, even with a valid fixture world
    assert hx.main(["execute", "--manifest", str(world.manifest_path)]) == 2
    assert world.ledger_lines() == [] and not (world.handoff / f"{world.nonce}.operation.json").exists()


@pytest.mark.parametrize("mutate, code", [
    (lambda m: m.update(skip_holder_check=True), "unknown_key"),
    (lambda m: m["artifact"].update(executor_sha256="0" * 64), "artifact_hash"),
    (lambda m: m["holders"][0].pop("ct"), "holder_schema"),
    (lambda m: m["plugin"]["files"].pop(), "payload_manifest"),
    (lambda m: m["plugin"]["files"][0].update(sha256="1" * 64), "payload_manifest"),
    (lambda m: m["services"].update(webui_health_url="http://10.0.0.5:80/health"), "service_schema"),
    (lambda m: m.update(final_result="../elsewhere.json"), "path_schema"),
])
def test_manifest_inputs_that_weaken_or_omit_bindings_fail_closed(world, mutate, code):
    mutate(world.manifest); world.save()
    with pytest.raises(hx.ManifestError) as err:
        hx.load_manifest(world.manifest_path, world.pins)
    assert err.value.code == code
    r = world.run()
    assert r["outcome"] == "refused" and r["reason"] == code
    assert world.ledger_lines() == [] and world.requests(0) == [] and world.final() is None


@pytest.mark.parametrize("cgroup, env, code", [
    ("0::/system.slice/hermes-webui.service\n", {"INVOCATION_ID": "x"}, "hosting_cgroup"),
    ("0::/system.slice/hermes-gateway.service\n", {"INVOCATION_ID": "x"}, "hosting_cgroup"),
    (CUTOVER_CGROUP, {}, "not_transient_service"),
    ("0::/user.slice/user-1000.slice/session-4.scope\n", {"INVOCATION_ID": "x"}, "not_transient_service"),
])
def test_independent_cgroup_guard(world, cgroup, env, code):
    with pytest.raises(hx.ManifestError) as err:
        hx.cgroup_guard(cgroup, env, world.pins)
    assert err.value.code == code
    r = hx.execute(world.manifest_path, pins=world.pins, host=world.host, cgroup_text=cgroup, environ=env)
    assert (r["outcome"], r["reason"]) == ("refused", code) and world.ledger_lines() == []


def test_stale_checkpoint_and_deadline_refuse_before_claiming_the_nonce(world):
    world.write_checkpoint(age=2000)
    assert world.run()["reason"] == "checkpoint_stale"
    world.write_checkpoint()
    world.manifest["deadline_utc"] = "2020-01-01T00:00:00+00:00"; world.save()
    assert world.run()["reason"] == "deadline_passed"
    assert not (world.handoff / f"{world.nonce}.operation.json").exists() and world.requests(0) == []


def test_dirty_checkout_refuses_durably_and_the_nonce_cannot_be_replayed(world):
    (world.checkout / "keep.txt").write_text("local edit\n")
    r = world.run()
    assert (r["outcome"], r["reason"], r["acceptance_passed"]) == ("refused", "dirty_checkout", False)
    assert world.final()["outcome"] == "refused" and world.final()["acceptance_passed"] is False
    (world.checkout / "keep.txt").write_text("unchanged tracked\n")
    (world.handoff / "CUTOVER-RUNTIME-ACCEPTANCE.json").unlink()
    assert world.run()["reason"] == "replay"
    assert world.ledger_lines() == [] and world.requests(0) == []


def test_symlinked_plugin_dir_and_foreign_owner_refuse(world):
    real = world.plugin.with_name("honcho-real"); world.plugin.rename(real); world.plugin.symlink_to(real)
    assert world.run()["reason"] == "symlink"
    world.plugin.unlink(); real.rename(world.plugin)
    world.manifest["nonce"] = world.nonce = os.urandom(16).hex(); world.write_checkpoint()
    world.manifest["owner"]["uid"] = os.getuid() + 1; world.save()
    assert world.run()["reason"] == "owner_mismatch"
    assert world.ledger_lines() == []


# ---- holders: identity, credentials, permits ---------------------------------------------------

def test_omitted_holder_is_refused(world):
    world.manifest["holders"] = []; world.save()
    r = world.run()
    assert (r["outcome"], r["reason"]) == ("refused", "undeclared_holder")
    assert world.requests(0) == [] and world.ledger_lines() == []


def test_recycled_or_drifted_holder_identity_refuses_without_http_or_signal(world):
    world.manifest["holders"][0]["ct"] = "ct:%.3f" % (float(world.holders[0]["ct"][3:]) - 50); world.save()
    r = world.run()
    assert (r["outcome"], r["reason"]) == ("refused", "holder_identity_changed")
    assert world.requests(0) == [] and world.procs[0].poll() is None


def test_wrong_holder_port_is_a_listener_mismatch(world):
    world.manifest["holders"][0]["port"] = world.port; world.save()
    assert world.run()["reason"] == "holder_listener_mismatch"
    assert world.requests(0) == []


@pytest.mark.parametrize("cfg, code", [
    ({"gated": True}, "holder_gated"),
    ({"withheld": True}, "credential_withheld"),
    ({"reject_auth": True}, "auth_failed"),
    ({"idle": None}, "permit_refused"),
    ({"idle": False}, "permit_refused"),
    ({"expire_after_prepare": True}, "permit_commit_failed"),
])
def test_holder_credentials_and_permits_fail_closed_before_any_stop(world, cfg, code):
    world.serve_cfg(0, **cfg)
    r = world.run()
    assert (r["outcome"], r["reason"], r["acceptance_passed"]) == ("aborted_before_switch", code, False)
    assert world.procs[0].poll() is None, "a refused holder is never signalled"
    assert world.ledger_lines() == [] and git(world.checkout, "rev-parse", "HEAD") == world.base
    fence = world.fence(0)
    assert fence is None or fence == {"committed": False, "permit_open": False}
    assert world.final()["outcome"] == "aborted_before_switch"


@_serves({"role": "desktop-serve"}, {"role": "serve"})
def test_second_holder_refusal_cancels_the_first_uncommitted_permit(world):
    world.serve_cfg(1, idle=None)
    r = world.run()
    assert r["reason"] == "permit_refused"
    assert world.fence(0) == {"committed": False, "permit_open": False}
    assert all(p.poll() is None for p in world.procs)
    rows = {row["pre"]["pid"]: row["action"] for row in r["holders"]}
    assert rows == {world.holders[0]["pid"]: "cancelled", world.holders[1]["pid"]: "refused"}


@_serves({"role": "desktop-serve"}, {"role": "serve"})
def test_partial_commit_is_recorded_honestly_and_nothing_is_stopped(world):
    world.serve_cfg(1, mangle_commit=True)
    r = world.run()
    assert (r["outcome"], r["reason"]) == ("aborted_before_switch", "permit_commit_failed")
    assert world.fence(0) == {"committed": True, "permit_open": False}
    assert world.fence(1) == {"committed": False, "permit_open": False}
    rows = {row["pre"]["pid"]: row["action"] for row in r["holders"]}
    assert rows == {world.holders[0]["pid"]: "committed_not_retired", world.holders[1]["pid"]: "cancelled"}
    assert all(p.poll() is None for p in world.procs) and world.ledger_lines() == []
    assert r["partial_retirement"] is True


# ---- gateway / WebUI quiescence -----------------------------------------------------------------

@pytest.mark.parametrize("active, code", [([0, 1, 1, 1, 1, 1, 1, 1], "gateway_busy"), ([None], "gateway_indeterminate")])
def test_busy_or_indeterminate_gateway_refuses_before_any_holder_action(world, active, code):
    world.cfg("gateway", {"systemctl": str(world.systemctl), "active": active})
    r = world.run()
    assert (r["outcome"], r["reason"]) == ("refused", code)
    assert world.requests(0) == [] or all(x.startswith("GET") for x in world.requests(0))
    assert world.ledger_lines() == [] and world.procs[0].poll() is None


def test_webui_nested_worker_refuses_before_any_stop(world):
    extra = subprocess.Popen([PY, "-c", "import signal; signal.pause()"])
    world.procs.append(extra)
    units = json.loads(world.units.read_text()); units[UI]["extra"] = [extra.pid]; world.units.write_text(json.dumps(units))
    (world.cg / "system.slice" / UI / "cgroup.procs").write_text(f"{units[UI]['pid']}\n{extra.pid}\n")
    r = world.run()
    assert (r["outcome"], r["reason"]) == ("refused", "webui_nested_workers")
    assert world.ledger_lines() == [] and world.fence(0) is None


def test_drain_ack_without_exit_never_stops_the_unit(world):
    world.cfg("gateway", {"systemctl": str(world.systemctl), "pause": "ignore"})
    gw = world.unit(GW)["pid"]
    r = world.run()
    assert (r["outcome"], r["reason"]) == ("aborted_before_switch", "gateway_drain_not_exited")
    assert world.ledger_lines() == [] and world.unit(GW)["pid"] == gw and psutil.pid_exists(gw)
    assert git(world.checkout, "rev-parse", "HEAD") == world.base


def test_respawned_gateway_generation_stops_the_cutover(world):
    world.cfg("gateway", {"systemctl": str(world.systemctl), "pause": "respawn"})
    old = world.unit(GW)["pid"]
    r = world.run()
    assert (r["outcome"], r["reason"]) == ("aborted_before_switch", "gateway_respawned")
    assert world.unit(GW)["pid"] not in (0, old) and ["stop", GW] not in world.ledger_lines()
    assert git(world.checkout, "rev-parse", "HEAD") == world.base


def test_update_lock_held_elsewhere_refuses_and_leaves_marker(world):
    fd = os.open(world.checkout / ".git" / "fixture-update.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    (world.home / ".update_in_progress").write_text("other updater\n")
    try:
        r = world.run()
    finally:
        os.close(fd)
    assert (r["outcome"], r["reason"]) == ("refused", "lock_held")
    assert (world.home / ".update_in_progress").read_text() == "other updater\n"
    assert world.requests(0) == [] and world.ledger_lines() == []


# ---- switch, start, verify, rollback, hold -------------------------------------------------------

def test_success_switches_exact_pair_verifies_fresh_runtime_and_resumes_post_start_once(world):
    keep, marker = meta(world.checkout / "keep.txt"), meta(world.marker)
    pyproject, notes = meta(world.plugin / "pyproject.toml"), meta(world.plugin / "user-notes.txt")
    old_gw, old_ui = world.unit(GW), world.unit(UI)
    r = world.run()
    assert (r["outcome"], r["acceptance_passed"]) == ("success", True), r
    final = world.final()
    assert final["outcome"] == "success" and final["acceptance_passed"] is True and final["nonce"] == world.nonce
    assert final["evidence"]["in_process_plugin_digest_api"] == "unavailable"
    assert final["evidence"]["honcho_sdk"] == "2.8.0" and final["evidence"]["lock_held_across_boot"] is True
    assert final["reconciliation"]["summary"] == "2 services updated and verified, 1 Desktop generations retired, 0 replacements observed"
    assert git(world.checkout, "rev-parse", "HEAD") == world.target
    assert meta(world.checkout / "keep.txt") == keep and meta(world.marker) == marker
    assert meta(world.plugin / "pyproject.toml") == pyproject and meta(world.plugin / "user-notes.txt") == notes
    assert all(sha256(world.plugin / f) == sha256(world.plugin_src / f) for f in world.pfiles)
    assert world.procs[0].wait(timeout=10) is not None
    gw, ui = world.unit(GW), world.unit(UI)
    assert gw["pid"] != old_gw["pid"] and gw["invocation"] != old_gw["invocation"] and ui["pid"] != old_ui["pid"]
    assert world.boot("gateway")["code_sha"] == world.target
    assert world.ledger_lines() == [["stop", GW], ["stop", UI], ["start", GW], ["start", UI]]
    assert Path((world.home / "pm-state" / "selected").read_text()) != world.old_venv
    assert not (world.home / ".update_in_progress").exists()
    assert world.hermes_calls() == [["cron", "resume", "fixturejob01", "--run-now"]]
    post = json.loads((world.handoff / f"{world.nonce}.post-start.json").read_text())
    assert post["command"][-4:] == ["cron", "resume", "fixturejob01", "--run-now"] and post["readback"]["enabled"] is True
    pids = [c["pid"] for c in r["children"]]
    assert len(pids) == len(set(pids)) and "run_agent" not in sys.modules
    stages = [json.loads(x)["stage"] for x in (world.handoff / f"{world.nonce}.journal.ndjson").read_text().splitlines()]
    assert stages.index("final") > stages.index("reconciliation") > stages.index("switch_done")
    again = hx.post_start(world.manifest_path, pins=world.pins, host=world.host)
    assert again["resumed"] is False and len(world.hermes_calls()) == 1


def test_pm_failure_rolls_back_payload_and_reports_new_environment_directory(world):
    before = {f: meta(world.plugin / f) for f in ("__init__.py", "pyproject.toml", "plugin.yaml", "user-notes.txt")}
    (world.home / "pm-state" / "fail-next-sync").write_text("")
    r = world.run()
    assert (r["outcome"], r["acceptance_passed"]) == ("rolled_back", False)
    assert r["failure"]["reason"] == "pm_sync_failed" and r["environment_directory_restored"] is False
    assert git(world.checkout, "rev-parse", "HEAD") == world.base
    assert {f: meta(world.plugin / f) for f in before} == before
    assert not (world.plugin / "provider.py").exists()
    assert list((world.handoff / f"{world.nonce}.backup" / "candidate-only").rglob("provider.py"))
    assert world.boot("gateway")["code_sha"] == world.base and world.unit(UI)["pid"]
    assert Path((world.home / "pm-state" / "selected").read_text()) != world.old_venv
    assert world.final()["outcome"] == "rolled_back" and world.hermes_calls() == []


def test_gateway_that_never_identifies_rolls_back(world):
    world.cfg("gateway", {"systemctl": str(world.systemctl), "crash_on": [world.target]})
    r = world.run()
    assert (r["outcome"], r["failure"]["reason"]) == ("rolled_back", "gateway_start_failed")
    assert git(world.checkout, "rev-parse", "HEAD") == world.base and world.boot("gateway")["code_sha"] == world.base
    assert world.hermes_calls() == []


def test_rollback_health_failure_is_rollback_failed_hold(world):
    world.cfg("gateway", {"systemctl": str(world.systemctl), "crash_on": [world.target, world.base]})
    r = world.run()
    assert (r["outcome"], r["failure"]["reason"], r["acceptance_passed"]) == ("rollback_failed", "gateway_start_failed", False)
    assert r["recovery"]["verified"] is False and world.final()["outcome"] == "rollback_failed"
    assert world.hermes_calls() == []


def test_webui_failure_after_healthy_new_gateway_is_hold_not_rollback(world):
    world.cfg("webui", {"port": world.port, "crash_on": [world.target]})
    r = world.run()
    assert (r["outcome"], r["reason"], r["acceptance_passed"]) == ("hold_after_start", "webui_start_failed", False)
    assert git(world.checkout, "rev-parse", "HEAD") == world.target and world.boot("gateway")["code_sha"] == world.target
    assert psutil.pid_exists(world.unit(GW)["pid"]) and world.hermes_calls() == []


def test_stale_boot_identity_with_healthy_services_is_hold(world):
    world.cfg("gateway", {"systemctl": str(world.systemctl), "identify_map": {world.target: world.base}})
    r = world.run()
    assert (r["outcome"], r["reason"]) == ("hold_after_start", "boot_identity_mismatch")
    assert git(world.checkout, "rev-parse", "HEAD") == world.target and world.hermes_calls() == []


def test_interruption_before_final_result_leaves_no_acceptance_and_no_trigger(world):
    def crash(_record):
        raise RuntimeError("injected interruption")
    with pytest.raises(RuntimeError):
        world.run(hooks={"before_final": crash})
    assert world.final() is None and (world.handoff / f"{world.nonce}.reconciliation.json").exists()
    assert hx.post_start(world.manifest_path, pins=world.pins, host=world.host) == {"resumed": False, "reason": "no_final_result"}
    assert world.run()["reason"] == "replay" and world.hermes_calls() == []


@pytest.mark.parametrize("job, final, code", [
    ({"repeat": {"times": 1, "completed": 1}}, {}, "job_spent"),
    ({"run_claim": {"at": "now"}}, {}, "job_claimed"),
    ({"enabled": True}, {}, "job_not_paused"),
    ({}, {"outcome": "rolled_back", "acceptance_passed": False}, "not_success"),
    ({}, {"services": "stale"}, "services_not_fresh"),
])
def test_post_start_refuses_premature_spent_claimed_or_unhealthy(world, job, final, code):
    world.set_job(**job)
    (world.handoff / f"{world.nonce}.operation.json").write_text(json.dumps({"nonce": world.nonce}))
    services = {"gateway": {"pid": world.unit(GW)["pid"], "invocation": world.unit(GW)["invocation"]},
                "webui": {"pid": world.unit(UI)["pid"], "invocation": world.unit(UI)["invocation"]}}
    if final.pop("services", None) == "stale":
        services["webui"]["invocation"] = "0" * 32
    record = {"schema": "honcho-exact-pair-result/1", "nonce": world.nonce, "outcome": "success",
              "acceptance_passed": True, "services": services, **final}
    (world.handoff / "CUTOVER-RUNTIME-ACCEPTANCE.json").write_text(json.dumps(record))
    r = hx.post_start(world.manifest_path, pins=world.pins, host=world.host)
    assert r == {"resumed": False, "reason": code} and world.hermes_calls() == []

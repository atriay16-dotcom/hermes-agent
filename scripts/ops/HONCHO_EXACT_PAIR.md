# One-time Honcho exact-pair cutover executor

`honcho_exact_pair.py` (+ `honcho_exact_pair_runtime.py`) installs ONLY the frozen pair into the existing
installation and proves the fresh runtime, or ends in a durable, truthful non-success outcome. It is an
operational artifact on branch `ops/honcho-exact-pair-executor-20261109`; the operational commit is never
installed. Pins (`LIVE_PINS`): core `a61e8dc5aa33818d2995db58e8aea76fe05b174f` / tree
`242a5bb62684000d596b212ef4916cc6cb657b27`, plugin `06664b491ab9a18edcf1dd48f68cf66e149483f8` / tree
`8d39010ccfecbfd6e8973fbf669ff1868ec7cc73` (64 files), post-start job `865e1487a1ad`, honcho-ai `2.8.0`.

Nothing here is armed. Live use belongs to the parent only after focused tests, independent review and
exact-current-artifact CI, with a fresh owner checkpoint and controller yield.

## Commands

```
python -I scripts/ops/honcho_exact_pair.py plan       --manifest M.json   # read-only preflight
python -I scripts/ops/honcho_exact_pair.py execute    --manifest M.json --apply
python -I scripts/ops/honcho_exact_pair.py post-start --manifest M.json   # recovery re-check only
```

`execute` refuses unless `--apply`, euid 0, `INVOCATION_ID` is set and its own cgroup leaf is
`hermes-honcho-cutover-*.service` with no `hermes-gateway.service` / `hermes-webui.service` segment.
Launch it outside every hosting unit with the existing transient-unit facility (no timer/daemon),
through the retained parent-owned SSH/sudo route:

```
sudo systemd-run --unit=hermes-honcho-cutover-<UTCSTAMP> --collect \
  --property=Type=oneshot --property=KillMode=process \
  -- <python> -I <operational checkout>/scripts/ops/honcho_exact_pair.py execute --manifest <M.json> --apply
```

The executor never depends on a gateway/WebUI tool connection; follow it with
`journalctl -u hermes-honcho-cutover-<UTCSTAMP>` and the files below.

## Manifest (`honcho-exact-pair/1`, every key required, unknown keys refused)

| key | meaning |
|---|---|
| `nonce` | 32-64 hex; claimed once as `<handoff>/<nonce>.operation.json` (O_EXCL, never repaired/reused) |
| `deadline_utc` | aware ISO time; refused after it (also re-checked before the switch) |
| `artifact` | `executor_sha256`, `runtime_sha256` of these two files as reviewed |
| `owner` | `user`, `uid`, `gid`, `home` of the installation owner (children run via `runuser -u`) |
| `profile_home`, `python` | HERMES_HOME for native children; non-runtime interpreter for owner children |
| `core` | `checkout` (live), `source_repo` (frozen clone), `commit`/`tree` (= pins), `baseline` |
| `plugin` | `live_dir`, `source_repo`, `commit`/`tree` (= pins), `files` [{path, sha256}] = the 64-file manifest |
| `services` | `gateway`, `webui` unit names (= pins) and `webui_health_url` `http://127.0.0.1:<port>/health` |
| `holders` | every Desktop serve generation: `pid`, `ct` (`update_pause_record.identity` spelling), `port`, `role` |
| `checkpoint` | owner quiet checkpoint `{nonce, created_at, controller_yielded: true}`, at most 15 min old |
| `handoff_dir`, `final_result` | durable evidence directory and result basename (default `CUTOVER-RUNTIME-ACCEPTANCE.json`) |
| `post_start` | `job_id` (= pin) and absolute `hermes_bin` |

No path is discovered by globbing; secrets are never inputs, arguments or log content. The Desktop
session token is read from each ungated loopback serve's own SPA page only after pid/start/uid/listener
checks, sent only in the `X-Hermes-Session-Token` header, and never stored.

## Sequence and failure semantics

1. **Preflight (read-only, `refused`)**: no symlinks, owner uid on checkout/plugin/profile/handoff, checkout
   HEAD = baseline with clean tracked tree (untracked work preserved), frozen trees/blobs, PM selected
   environment, unit MainPID/InvocationID/Restart/KillMode/ControlGroup, gateway `identify` boot sha =
   baseline, every declared holder's identity and loopback listener, every process mapping bytes under the
   checkout or selected environment accounted for (service cgroups or declared holders), paused unused job.
2. **Lock**: an owner child holds native `UpdateLock(install_root=checkout)` for the whole run; switch
   children adopt it through `HERMES_UPDATE_HANDOFF_PID`. Held elsewhere -> `lock_held`. The frozen commit
   is fetched from the frozen clone if absent and its tree verified.
3. **Quiet (`refused`)**: gateway `active-work` zero on two consecutive samples within a bounded wait
   (busy -> `gateway_busy`, unanswered -> `gateway_indeterminate`); the WebUI cgroup lists only its
   MainPID (no read-only WebUI busy API exists; cgroup + owner checkpoint is the truthful probe).
4. **Desktop holders (`aborted_before_switch`)**: health must be ungated with baseline commit, served
   credential present, authenticated identity pid/role match; prepare every permit, then commit every
   permit inside the 30 s fence window. Any refusal/indeterminate idle/expiry cancels every uncommitted
   permit. Committed fences cannot be cancelled: they are recorded `committed_not_retired`
   (`partial_retirement: true`). Only after all commits is each exact generation (pidfd) sent SIGTERM and
   its exit awaited; no SIGKILL, no deferred-busy override.
5. **Gateway, then WebUI (`aborted_before_switch`, stopped units restarted on old code)**: re-check
   active-work, `pause-for-update`, wait for that exact generation to exit; a survivor is reported, never
   stopped; a respawned generation stops the cutover. Only after exit does `systemctl stop` disarm the
   respawn. WebUI: checkpoint and cgroup re-checked, then `systemctl stop`.
6. **Switch (point of no return)**: fresh owner children: detached checkout of the exact core sha;
   frozen plugin blobs written (unchanged bytes untouched, changed files in place, new files atomically;
   baseline bytes/metadata backed up under `<nonce>.backup/`); PM sync with explicit Members and no
   eviction; the selected environment must change and carry honcho-ai 2.8.0.
7. **Start**: gateway via its unchanged unit; it must answer `identify` as a new pid/InvocationID that
   started after the switch. Failure -> rollback. Then WebUI: new pid/InvocationID + `/health` 200.
   WebUI failure or a boot sha other than the pin with healthy services -> `hold_after_start` (no rollback).
8. **Verify**: both fresh processes map the new selected environment and not the old; no process maps the
   old environment; disk hashes identical before start and after boot; lock held across boot.
9. **Rollback** (switch or gateway-start failure): stop started units, baseline checkout, restore plugin
   baseline bytes/mode/mtime and archive candidate-only files, PM sync again, start both units, verify old
   boot sha. `rolled_back` or `rollback_failed`; the original failure is kept. The environment is a NEW
   directory: `environment_directory_restored: false` (105 versions, not the identical directory).

Every outcome after the nonce claim writes `<nonce>.reconciliation.json`, then the final result (atomic,
fsynced) with `outcome` in `refused | aborted_before_switch | rolled_back | rollback_failed |
hold_after_start | success`, `acceptance_passed` true only for `success`, and `release_gate`. A
journal `<nonce>.journal.ndjson` records each stage. Interruption before the final write leaves no
final result and therefore no trigger.

## Evidence limits

There is no in-process plugin digest API (`in_process_plugin_digest_api: "unavailable"`). Runtime
evidence is: fresh pid/start/InvocationID, gateway boot `code_sha` from the answering process, lock
held across boot, both processes mapping the new selected environment, honcho-ai 2.8.0 present in it,
and frozen bytes unchanged across the boot window. Process `environ`/`argv` are never read. Desktop
rows end `counted_as_updated: false`; the summary counts services verified, generations retired and
replacements observed, never four updated processes.

## Post-start trigger

Only after a durable `success` final result, with both services still the recorded fresh generations,
gateway boot sha = pin and WebUI healthy, and the job still `enabled: false`, `next_run_at: null`,
`repeat.completed == 0`, no run/fire claim: the executor claims `<nonce>.post-start.json` (once) and runs
`hermes cron resume 865e1487a1ad --run-now` as the owner, storing the command and a registry readback.
`rearm_oneshot` would reset a spent occurrence, so these checks are the only guard; any failure is HOLD.
The job itself must detect already-progressing EXIT work and avoid duplicate dispatch.

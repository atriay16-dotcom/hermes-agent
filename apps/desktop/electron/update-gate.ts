'use strict'

/**
 * update-gate.ts
 *
 * Pure, dependency-injected gate that parks local backend spawns while an
 * in-app update is running (#73822, #50238).
 *
 * Two independent signals mean "an update owns the venv right now":
 *
 *  - the on-disk marker (`HERMES_HOME/.hermes-update-in-progress`), written
 *    by the updater — and by the desktop itself just before hand-off — and
 *  - the in-process `updateInFlight` flag, true for the whole
 *    `applyUpdates()` critical section.
 *
 * The marker alone is NOT enough (#73822): `applyUpdates` kills its own
 * backend early (`releaseBackendLock`) but only writes the marker AFTER the
 * Windows venv-blocker scan. Killing the backend drops the renderer's
 * WebSocket, the renderer reconnects within ~1s, and a marker-only gate
 * happily spawns a fresh backend inside the update's own critical section —
 * which `scanVenvBlockers` then reports as a blocker, aborting every update
 * attempt forever. Consulting the flag closes that window. On the success
 * path the marker is written BEFORE the flag clears in `applyUpdates`'
 * `finally`, so there is no instant where both signals are false and a
 * waiter could slip through mid-update.
 */

export type UpdateGateReason = 'marker' | 'update-in-flight' | null

export interface UpdateGateDeps {
  /** True when a live on-disk update marker exists (see update-marker.ts). */
  hasLiveMarker: () => boolean
  /** True while this process is inside applyUpdates()' critical section. */
  isUpdateInFlight: () => boolean
}

/** Why the gate is closed right now, or null when it is open. */
export function updateGateReason(deps: UpdateGateDeps): UpdateGateReason {
  if (deps.hasLiveMarker()) {
    return 'marker'
  }

  if (deps.isUpdateInFlight()) {
    return 'update-in-flight'
  }

  return null
}

export type UpdateClearanceOutcome = 'clear' | 'finished' | 'timeout'

export interface WaitForUpdateClearanceOptions {
  timeoutMs: number
  pollMs: number
  /** Invoked once per poll while parked (boot progress / logging). */
  onWaitTick?: (reason: Exclude<UpdateGateReason, null>) => void | Promise<void>
  now?: () => number
  sleep?: (ms: number) => Promise<void>
}

/**
 * Park until no update signal remains, or the deadline passes.
 *
 * Returns 'clear' when the gate was already open (no wait happened),
 * 'finished' when it opened during the wait, and 'timeout' when the deadline
 * expired with the gate still closed (callers proceed anyway — matching the
 * long-standing marker-gate behavior, since a wedged updater must not brick
 * the app forever).
 */
export async function waitForUpdateClearance(
  deps: UpdateGateDeps,
  options: WaitForUpdateClearanceOptions
): Promise<UpdateClearanceOutcome> {
  const now = options.now || Date.now
  const sleep = options.sleep || (ms => new Promise<void>(r => setTimeout(r, ms)))

  let reason = updateGateReason(deps)

  if (!reason) {
    return 'clear'
  }

  const deadline = now() + options.timeoutMs

  while (reason && now() < deadline) {
    if (options.onWaitTick) {
      await options.onWaitTick(reason)
    }

    await sleep(options.pollMs)
    reason = updateGateReason(deps)
  }

  return reason ? 'timeout' : 'finished'
}

/**
 * Ids with an IN-PROCESS managed SSH update/recovery, deduped (Sep 1 Windows
 * race) — exactly what `before-quit` awaits, so exactly what a local hand-off
 * must refuse to race: the detached updater's fixed exit deadline is what a
 * quit-wait consumes. Durable journal records are deliberately excluded: none
 * is awaited, and one only clears on a fully successful restore, so blocking
 * on it would lock client updates out while a remote stays down.
 */
export function pendingManagedRemoteWork(updateIds: Iterable<string>, recoveryIds: Iterable<string>): string[] {
  return [...new Set([...updateIds, ...recoveryIds])]
}

export interface LocalHandoffSignals {
  isUpdateInFlight: () => boolean
  isQuittingForHandoff: () => boolean
}

/**
 * Why a sibling update must not start while the local/client hand-off owns the
 * desktop's exit deadline, or null when it may. Neither signal subsumes the
 * other: applyUpdates clears `updateInFlight` in its `finally` while the dwell
 * timer still has `app.quit()` scheduled, so `isQuittingForHandoff` is all
 * that is left through the dwell.
 */
export function localHandoffBlock(signals: LocalHandoffSignals): 'update-in-flight' | 'quitting-for-handoff' | null {
  if (signals.isUpdateInFlight()) {
    return 'update-in-flight'
  }

  if (signals.isQuittingForHandoff()) {
    return 'quitting-for-handoff'
  }

  return null
}

export interface UpdateLifecycleOwner extends LocalHandoffSignals {
  setUpdateInFlight: (inFlight: boolean) => void
}

export interface UpdateLifecycleReservation {
  ok: boolean
  blockedBy: 'update-in-flight' | 'quitting-for-handoff' | null
  release: () => void
}

/**
 * Take ownership of the update lifecycle, synchronously — no await may sit
 * between the check and the claim, or two entry points both read "free".
 *
 * A refusal claims nothing, so its `release` is a no-op: callers release from a
 * `finally` that also runs on the refusal path, and clearing there would hand
 * the gate away from whoever actually owns it. `release` is likewise one-shot,
 * so a stale release cannot clear a later owner's reservation.
 */
export function reserveUpdateLifecycle(owner: UpdateLifecycleOwner): UpdateLifecycleReservation {
  const blockedBy = localHandoffBlock(owner)

  if (blockedBy) {
    return { ok: false, blockedBy, release: () => undefined }
  }

  owner.setUpdateInFlight(true)
  let released = false

  return {
    ok: true,
    blockedBy: null,
    release: () => {
      if (!released) {
        released = true
        owner.setUpdateInFlight(false)
      }
    }
  }
}

/**
 * Per-row refusal for a fleet row that cannot start right now, or null when it
 * may proceed. Shaped like the batch's other skipped rows so refusing one row
 * leaves every unrelated row's outcome intact.
 */
export function localHandoffRowRefusal<B extends object>(base: B, signals: LocalHandoffSignals) {
  const block = localHandoffBlock(signals)

  return block
    ? { ...base, ok: false, skipped: true, reason: block, detail: 'A local/client update hand-off is in progress.' }
    : null
}

/**
 * Run one remote fleet row's update behind the hand-off gate in BOTH phases:
 * dialing a cold backend takes seconds, and a hand-off committed by another
 * entry point during that await would leave the POST racing the updater's
 * fixed exit deadline. The post-dial recheck is the whole point of this seam.
 */
export async function guardedRemoteRowUpdate<B extends object, D, R>(
  base: B,
  signals: LocalHandoffSignals,
  dial: () => Promise<D>,
  post: (dialed: D) => Promise<R>
): Promise<R | NonNullable<ReturnType<typeof localHandoffRowRefusal<B>>>> {
  const beforeDial = localHandoffRowRefusal(base, signals)

  if (beforeDial) {
    return beforeDial
  }

  const dialed = await dial()
  const beforePost = localHandoffRowRefusal(base, signals)

  return beforePost || (await post(dialed))
}

/**
 * Dispatch remote connections concurrently, then local ones only once every
 * remote has settled — local's fixed exit deadline must not race open-ended
 * remote work, even when local appears first in the registry.
 *
 * One row's throw must not decide another row's fate, so a rejection becomes
 * that row's own result via `onRowError` rather than rejecting the batch: the
 * row callback builds its `base`/eligibility outside its own try/catch, and
 * re-throwing from here would drop every sibling row's outcome (and the whole
 * update-all IPC response) on the floor. Rows come back in the caller's order.
 */
export async function dispatchUpdatesRemoteFirst<C extends { id: string; kind: string }, R>(
  connections: C[],
  dispatch: (connection: C) => Promise<R>,
  onRowError: (connection: C, error: unknown) => R
): Promise<R[]> {
  const results = new Array<R>(connections.length)
  const runPhase = (localPhase: boolean) =>
    Promise.all(
      connections.map(async (connection, index) => {
        if ((connection.kind === 'local') !== localPhase) {
          return
        }

        try {
          results[index] = await dispatch(connection)
        } catch (error) {
          results[index] = onRowError(connection, error)
        }
      })
    )

  await runPhase(false)
  await runPhase(true)

  return results
}

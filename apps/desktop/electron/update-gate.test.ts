/**
 * Tests for electron/update-gate.ts — the update mutual-exclusion gate that
 * parks local backend spawns while an in-app update is running.
 *
 * The regression this guards (#73822): applyUpdates kills its own backend
 * BEFORE the Windows venv-blocker scan but writes the on-disk marker AFTER
 * it. A marker-only gate therefore let the renderer's reconnect spawn a
 * fresh backend inside the update's own critical section, which the scan
 * reported as a blocker — aborting every Desktop update attempt on Windows.
 * The gate must consult the in-process updateInFlight flag as well.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  dispatchUpdatesRemoteFirst,
  guardedRemoteRowUpdate,
  localHandoffBlock,
  localHandoffRowRefusal,
  pendingManagedRemoteWork,
  reserveUpdateLifecycle,
  updateGateReason,
  waitForUpdateClearance
} from './update-gate'

function deps(marker: boolean, inFlight: boolean) {
  return {
    hasLiveMarker: () => marker,
    isUpdateInFlight: () => inFlight
  }
}

// ---------------------------------------------------------------------------
// updateGateReason
// ---------------------------------------------------------------------------

test('gate open when neither marker nor flag is set', () => {
  assert.equal(updateGateReason(deps(false, false)), null)
})

test('marker alone closes the gate', () => {
  assert.equal(updateGateReason(deps(true, false)), 'marker')
})

test('updateInFlight alone closes the gate (#73822 — the pre-marker window)', () => {
  assert.equal(updateGateReason(deps(false, true)), 'update-in-flight')
})

test('marker wins as the reported reason when both are set', () => {
  assert.equal(updateGateReason(deps(true, true)), 'marker')
})

// ---------------------------------------------------------------------------
// waitForUpdateClearance
// ---------------------------------------------------------------------------

test('returns clear immediately without sleeping when the gate is open', async () => {
  let slept = 0

  const outcome = await waitForUpdateClearance(deps(false, false), {
    pollMs: 10,
    sleep: async () => {
      slept += 1
    },
    timeoutMs: 1000
  })

  assert.equal(outcome, 'clear')
  assert.equal(slept, 0)
})

test('parks on the in-flight flag and finishes when it clears', async () => {
  // Simulates the #73822 sequence: the reconnect arrives while updateInFlight
  // is true and no marker exists yet; the flag clears (abort path finally)
  // and the waiter proceeds.
  let inFlight = true
  let ticks = 0

  const outcome = await waitForUpdateClearance(
    { hasLiveMarker: () => false, isUpdateInFlight: () => inFlight },
    {
      onWaitTick: reason => {
        ticks += 1
        assert.equal(reason, 'update-in-flight')

        if (ticks >= 3) {
          inFlight = false
        }
      },
      pollMs: 1,
      sleep: async () => {},
      timeoutMs: 10_000
    }
  )

  assert.equal(outcome, 'finished')
  assert.equal(ticks, 3)
})

test('parks across the flag→marker handoff without a gap', async () => {
  // Success path: the marker is written (main.ts:2936) BEFORE applyUpdates'
  // finally clears the flag, so a waiter that arrived during the scan stays
  // parked through the transition instead of slipping through.
  let inFlight = true
  let marker = false
  let ticks = 0
  const reasons: string[] = []

  const outcome = await waitForUpdateClearance(
    { hasLiveMarker: () => marker, isUpdateInFlight: () => inFlight },
    {
      onWaitTick: reason => {
        ticks += 1
        reasons.push(reason)

        if (ticks === 2) {
          marker = true // updater hand-off: marker written first…
        }

        if (ticks === 3) {
          inFlight = false // …then the flag clears; marker still holds the gate
        }

        if (ticks === 5) {
          marker = false // updater finished
        }
      },
      pollMs: 1,
      sleep: async () => {},
      timeoutMs: 10_000
    }
  )

  assert.equal(outcome, 'finished')
  assert.deepEqual(reasons, ['update-in-flight', 'update-in-flight', 'marker', 'marker', 'marker'])
})

test('returns timeout when the gate never opens', async () => {
  let clock = 0

  const outcome = await waitForUpdateClearance(deps(true, false), {
    now: () => clock,
    pollMs: 10,
    sleep: async ms => {
      clock += ms
    },
    timeoutMs: 50
  })

  assert.equal(outcome, 'timeout')
})

// ---------------------------------------------------------------------------
// Local hand-off vs managed remote work (Sep 1 Windows abort)
//
// The detached updater starts a FIXED desktop-exit deadline, while before-quit
// deliberately waits for managed SSH update/recovery operations. If the two
// overlap, the wait eats the deadline and the client update aborts. The four
// invariants below are what the production entry points (applyUpdates,
// requestManagedSshUpdate, the fleet URL rows, handOffWindowsBootstrapRecovery)
// consult to keep them apart.
// ---------------------------------------------------------------------------

function signals(inFlight: boolean, quitting: boolean) {
  return {
    isUpdateInFlight: () => inFlight,
    isQuittingForHandoff: () => quitting
  }
}

// INV-PENDING-SET: the hand-off refusal set is the union of the two ACTIVE
// operation maps only. Durable journal ids are not an input here on purpose —
// nothing awaits them at before-quit, so they cannot consume the deadline.
test('pending managed work is the deduped union of active update and recovery ids', () => {
  const cases: Array<{ updates: string[]; recoveries: string[]; expected: string[] }> = [
    { updates: [], recoveries: [], expected: [] },
    { updates: ['office'], recoveries: [], expected: ['office'] },
    { updates: [], recoveries: ['lab'], expected: ['lab'] },
    { updates: ['office'], recoveries: ['office', 'lab'], expected: ['lab', 'office'] }
  ]

  for (const { updates, recoveries, expected } of cases) {
    assert.deepEqual(
      pendingManagedRemoteWork(updates, recoveries).sort(),
      expected,
      JSON.stringify({ updates, recoveries })
    )
  }
})

// INV-LIFECYCLE: both lifecycle flags block a sibling update, and neither
// subsumes the other. The third row is the post-finally dwell — applyUpdates
// has already cleared updateInFlight in its `finally` while the dwell timer
// still has app.quit() scheduled, so only isQuittingForHandoff is left.
test('a sibling update is refused for either lifecycle flag, including the post-finally dwell', () => {
  const cases: Array<{ inFlight: boolean; quitting: boolean; expected: string | null }> = [
    { inFlight: false, quitting: false, expected: null },
    { inFlight: true, quitting: false, expected: 'update-in-flight' },
    { inFlight: false, quitting: true, expected: 'quitting-for-handoff' },
    { inFlight: true, quitting: true, expected: 'update-in-flight' }
  ]

  for (const { inFlight, quitting, expected } of cases) {
    assert.equal(localHandoffBlock(signals(inFlight, quitting)), expected, JSON.stringify({ inFlight, quitting }))
  }
})

// INV-ROW-SHAPE: a refused fleet row is reported per-row (skipped, with the
// caller's base fields intact) rather than thrown, so one refusal cannot
// vanish the unrelated rows of the same batch. Re-evaluated per call because
// the URL row must recheck after awaiting its descriptor.
test('a refused fleet row keeps its base fields and reports per-row, and rechecks after an await', async () => {
  const base = { connectionId: 'url-box', label: 'URL box', kind: 'url' }

  assert.equal(localHandoffRowRefusal(base, signals(false, false)), null)

  let quitting = false
  const live = { isUpdateInFlight: () => false, isQuittingForHandoff: () => quitting }

  // Phase 1: the row is clear, so it proceeds into descriptor resolution.
  assert.equal(localHandoffRowRefusal(base, live), null)

  // A hand-off commits while the descriptor await is outstanding.
  await Promise.resolve()
  quitting = true

  // Phase 2 (immediately before the POST) must catch it.
  assert.deepEqual(localHandoffRowRefusal(base, live), {
    connectionId: 'url-box',
    label: 'URL box',
    kind: 'url',
    ok: false,
    skipped: true,
    reason: 'quitting-for-handoff',
    detail: 'A local/client update hand-off is in progress.'
  })
})

// ---------------------------------------------------------------------------
// dispatchUpdatesRemoteFirst
// ---------------------------------------------------------------------------

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, resolve, reject }
}

// INV-ORDER: remotes fan out together, local starts only once every remote has
// settled (even when local is listed first), and the caller gets its own row
// order back with each row's own outcome — including a failed remote row and a
// skipped cloud row.
test('local waits for every remote to settle while remotes run concurrently, preserving row order and per-row outcomes', async () => {
  const office = deferred<string>()
  const lab = deferred<string>()
  const started: string[] = []

  const results = dispatchUpdatesRemoteFirst(
    [
      { id: 'local', kind: 'local' },
      { id: 'office', kind: 'ssh' },
      { id: 'cloud', kind: 'cloud' },
      { id: 'lab', kind: 'url' }
    ],
    connection => {
      started.push(connection.id)

      if (connection.kind === 'local') {
        return Promise.resolve('local-ok')
      }

      if (connection.kind === 'cloud') {
        return Promise.resolve('cloud-skipped')
      }

      return connection.id === 'office' ? office.promise : lab.promise
    },
    (connection, error) => `${connection.id}-failed:${String(error)}`
  )

  await Promise.resolve()
  await Promise.resolve()

  // Every remote row is already in flight; local has not been touched.
  assert.deepEqual([...started].sort(), ['cloud', 'lab', 'office'])

  office.resolve('office-failed')
  lab.resolve('lab-ok')

  assert.deepEqual(await results, ['local-ok', 'office-failed', 'cloud-skipped', 'lab-ok'])
  assert.equal(started[started.length - 1], 'local')
})

// INV-ROW-ISOLATION: one row throwing must not decide another row's fate. The
// row callback builds `base`/`eligibility` OUTSIDE its own try/catch, so an
// unexpected throw there is reachable in production — and re-throwing it out of
// the dispatcher loses every sibling row's outcome and rejects the whole
// update-all IPC. Row failure is handled explicitly, at this boundary.
test('a throwing remote row is reported as its own row, leaving every sibling row intact', async () => {
  const started: string[] = []

  const results = await dispatchUpdatesRemoteFirst(
    [
      { id: 'local', kind: 'local' },
      { id: 'office', kind: 'ssh' },
      { id: 'lab', kind: 'url' }
    ],
    async connection => {
      started.push(connection.id)

      if (connection.id === 'office') {
        throw new Error('registry read blew up')
      }

      return `${connection.id}-ok`
    },
    (connection, error) => `${connection.id}-failed:${(error as Error).message}`
  )

  assert.deepEqual(results, ['local-ok', 'office-failed:registry read blew up', 'lab-ok'])
  assert.deepEqual([...started].sort(), ['lab', 'local', 'office'])
})

// ---------------------------------------------------------------------------
// reserveUpdateLifecycle — who owns the desktop's exit deadline
// ---------------------------------------------------------------------------

function lifecycle(initial: { inFlight?: boolean; quitting?: boolean } = {}) {
  const state = { inFlight: initial.inFlight === true, quitting: initial.quitting === true }

  return {
    state,
    owner: {
      isUpdateInFlight: () => state.inFlight,
      isQuittingForHandoff: () => state.quitting,
      setUpdateInFlight: (inFlight: boolean) => {
        state.inFlight = inFlight
      }
    }
  }
}

// INV-OWNERSHIP: a second entry point (the Windows bootstrap recovery hand-off
// is the live one) must not seize a reservation another hand-off already holds,
// and its `finally` release must not clear the owner's flag — that would open
// the gate mid-update and let a second updater spawn over the same install.
test('a reservation is refused while another hand-off owns it, and refusing never clears the owner', () => {
  const { state, owner } = lifecycle({ inFlight: true })

  const reservation = reserveUpdateLifecycle(owner)

  assert.equal(reservation.ok, false)
  assert.equal(reservation.blockedBy, 'update-in-flight')

  // main.ts releases in a `finally` that also runs on the refusal path.
  reservation.release()

  assert.equal(state.inFlight, true, 'the owning hand-off is still in flight')
})

// INV-DWELL: applyUpdates clears updateInFlight in its `finally` while the dwell
// timer still has app.quit() scheduled. updateInFlight alone therefore reads
// "free" during the dwell; isQuittingForHandoff is the only signal left.
test('a reservation is refused through the post-finally dwell', () => {
  const { state, owner } = lifecycle({ inFlight: false, quitting: true })

  const reservation = reserveUpdateLifecycle(owner)

  assert.equal(reservation.ok, false)
  assert.equal(reservation.blockedBy, 'quitting-for-handoff')
  assert.equal(state.inFlight, false, 'a refused reservation claims nothing')
})

// INV-RELEASE-ONCE: a stale release (double `finally`, retained closure) must
// not clear the NEXT owner's reservation.
test('release is idempotent, so a stale release cannot clear the next owner', () => {
  const { state, owner } = lifecycle()

  const first = reserveUpdateLifecycle(owner)

  assert.equal(first.ok, true)
  assert.equal(state.inFlight, true)

  first.release()
  assert.equal(state.inFlight, false)

  const second = reserveUpdateLifecycle(owner)

  assert.equal(second.ok, true)
  first.release()

  assert.equal(state.inFlight, true, "the second owner's reservation survives the stale release")
})

// ---------------------------------------------------------------------------
// guardedRemoteRowUpdate — the URL row's two-phase refusal
// ---------------------------------------------------------------------------

// INV-REMOTE-PHASE: a hand-off committed DURING the dial's await must stop the
// row before it POSTs. Driven through a deferred dial, which is the real
// production await (ensureRegistryBackend behind the dial claim).
test('a hand-off committed during the dial refuses the row before it POSTs', async () => {
  const { state, owner } = lifecycle()
  const dial = deferred<{ port: number }>()
  const posted: unknown[] = []

  const row = guardedRemoteRowUpdate(
    { connectionId: 'lab', kind: 'url' },
    owner,
    () => dial.promise,
    async descriptor => {
      posted.push(descriptor)

      return { connectionId: 'lab', kind: 'url', ok: true }
    }
  )

  // The hand-off commits while the dial is still outstanding.
  state.inFlight = true
  dial.resolve({ port: 8765 })

  assert.deepEqual(await row, {
    connectionId: 'lab',
    kind: 'url',
    ok: false,
    skipped: true,
    reason: 'update-in-flight',
    detail: 'A local/client update hand-off is in progress.'
  })
  assert.deepEqual(posted, [], 'the POST never raced the updater')
})

test('a row refused before the dial never dials at all, and an unblocked row posts its descriptor', async () => {
  const { state, owner } = lifecycle({ quitting: true })
  let dials = 0

  const refused: any = await guardedRemoteRowUpdate(
    { connectionId: 'lab', kind: 'url' },
    owner,
    async () => {
      dials += 1

      return { port: 8765 }
    },
    async () => ({ connectionId: 'lab', kind: 'url', ok: true })
  )

  assert.equal(refused.reason, 'quitting-for-handoff')
  assert.equal(dials, 0)

  state.quitting = false

  const allowed = await guardedRemoteRowUpdate(
    { connectionId: 'lab', kind: 'url' },
    owner,
    async () => {
      dials += 1

      return { port: 8765 }
    },
    async descriptor => ({ connectionId: 'lab', kind: 'url', ok: true, detail: `posted:${descriptor.port}` })
  )

  assert.deepEqual(allowed, { connectionId: 'lab', kind: 'url', ok: true, detail: 'posted:8765' })
  assert.equal(dials, 1)
})

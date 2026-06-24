# VDB Roadmap: From In-Memory Index to Production Database

This document is both a **plan** and a **learning guide**. Each phase explains the
database fundamental it teaches, then gives a concrete implementation plan tied to
the current codebase (`include/`, `src/`).

## Where we are today

`HNSWIndex` is a correct, in-memory ANN index. `VDB` is a thin wrapper over it.
The data lives in `std::vector<Node> node_pool_` and is lost on process exit.

A *database* adds four guarantees on top of an index — the classic **ACID** properties:

| Property        | Meaning here                                          | Phase |
|-----------------|-------------------------------------------------------|-------|
| **Durability**  | Committed writes survive a crash / restart            | 2     |
| **Atomicity**   | An operation fully applies or not at all              | 2     |
| **Isolation**   | Concurrent ops don't corrupt each other               | 3     |
| **Consistency** | Invariants (graph connectivity, ID maps) always hold  | 1–3   |

Recommended order: **Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5.**
Phase 1 is a prerequisite for the WAL (you need stable logical IDs to log against),
and durability (Phase 2) should exist before you invest in concurrency (Phase 3).

---

## Phase 1 — Identity & Lifecycle

**Fundamental:** A database never exposes physical storage offsets as the user key.
A layer of *indirection* between logical keys and physical location is what lets you
delete, move, and compact data underneath callers without breaking their references.

### 1.1 External ↔ internal ID mapping

Today callers get back an `InternalId` (= index into `node_pool_`). That's a physical
offset. Introduce a stable, user-facing `ExternalId` (already typedef'd in
`hnswindex.h`).

- Add to `HNSWIndex` (or better, a new owning layer in `VDB`):
  ```cpp
  std::unordered_map<ExternalId, InternalId> ext_to_int_;
  std::vector<ExternalId>                    int_to_ext_;   // indexed by InternalId
  ExternalId                                 next_ext_id_ = 1;
  ```
- `insert` returns an `ExternalId`. `get` / `search` translate at the boundary:
  internal IDs are used *only* inside the graph; the public API speaks `ExternalId`.
- Decide ownership: cleanest is to let **`VDB`** own the ID maps and metadata, and keep
  `HNSWIndex` purely about graph topology over `InternalId`. This separation pays off
  in every later phase.

### 1.2 Deletes via tombstones

You cannot cleanly excise a node from an HNSW graph — its neighbor links are load-bearing
for connectivity. Standard approach: **mark, then reclaim later.**

- Add a `std::vector<bool> deleted_` (or a `Node` flag).
- `search` skips deleted nodes when collecting the final result set, but **still traverses
  through them** so the graph stays navigable.
- Remove the `ExternalId` from `ext_to_int_` immediately so the key reads as gone.

**Fundamental:** "delete = mark now, reclaim later" is how almost every storage engine
works (Postgres dead tuples + VACUUM, LSM-tree tombstones + compaction).

### 1.3 Updates

`update(ext_id, new_vec)` = tombstone the old internal node + insert the new vector under
the **same** `ExternalId`. Atomicity of this pair matters once the WAL exists (Phase 2).

### 1.4 Compaction (can defer to after Phase 2)

When tombstones exceed a threshold, rebuild the graph from live nodes into a fresh
`node_pool_`, remapping internal IDs. External IDs are unaffected — that's the whole point
of the indirection.

**Deliverable:** `insert/get/search/delete/update` all keyed by `ExternalId`; tests for
delete-then-search and update semantics.

---

## Phase 2 — Durability: Write-Ahead Log + Snapshots

This is the core of "production database." Read this phase slowly.

### 2.0 The mental model

Two failure-proofing mechanisms working together:

1. **WAL (Write-Ahead Log):** an append-only file. *Before* mutating in-memory state,
   you append a record describing the change and flush it to disk. If you crash, you
   **replay** the log on startup to reconstruct state.
2. **Snapshot (checkpoint):** periodically serialize the whole index to disk so recovery
   doesn't have to replay the log since the beginning of time. After a snapshot, the old
   WAL can be truncated.

Recovery = **load newest snapshot, then replay the WAL records that came after it.**

The golden rule — and the literal meaning of *write-ahead* — is:

> **Log the intent durably BEFORE you change the in-memory data.**

If you change memory first and crash before logging, the write is lost but the user thinks
it succeeded. Log first, and the worst case is a logged-but-unapplied record, which replay
fixes. Ordering is everything.

### 2.1 On-disk WAL format

Design a **self-describing, append-only, checksummed** record stream. Concretely:

**File header (written once, at file creation):**
```
magic      : u32   = 0x56_44_42_57  ("VDBW")
version    : u16   = 1
dim        : u32   // guards against opening a DB with mismatched config
metric     : u8    // 0 = L2, 1 = IP, 2 = cosine ...
reserved   : pad to 32 bytes
```

**Record framing (repeated):**
```
length     : u32   // byte length of [type .. payload], not incl. crc
crc32c     : u32   // checksum over [length .. payload]
type       : u8    // see record types below
lsn        : u64   // Log Sequence Number, monotonically increasing
payload    : variable, depends on type
```

**Record types (payloads):**
```
INSERT (1): ext_id:u64, vec:f32[dim]
DELETE (2): ext_id:u64
UPDATE (3): ext_id:u64, vec:f32[dim]        // = delete + insert, one atomic record
CHECKPOINT (4): snapshot_id:u64             // marks "snapshot N is durable up to here"
```

Why each field exists — the lessons:

- **length prefix** → you can frame records without parsing them; lets you skip/scan.
- **crc32c** → detects **torn writes**. A crash mid-`write()` can leave a half-written
  record. On replay, a bad checksum means "this record and everything after it is garbage";
  you stop replaying there. *This is how you tell a clean tail from corruption.* (Use
  hardware CRC32C; it's a single instruction on modern x86/ARM.)
- **lsn (Log Sequence Number)** → a monotonic counter naming every write. It's the
  vocabulary for "how far has durability progressed" and ties the WAL to snapshots
  (a snapshot records "I contain all changes up to LSN = N").
- **type** → makes the log a replayable operation journal, not just data.

### 2.2 The write path (the fsync discipline)

For each mutating op (`insert`, `delete`, `update`):

1. Serialize the record into a buffer.
2. `write()` it to the WAL fd (appends to OS page cache).
3. **`fsync()` (or `fdatasync`) the WAL fd** — this is the durability point. The call
   returns only once the bytes are on stable storage.
4. *Only now* apply the change to `node_pool_` / ID maps in memory.
5. Return success to the caller.

**Fundamental — why `fsync` is the whole game:** `write()` only copies into the OS page
cache; the data is still in volatile RAM and a power cut loses it. `fsync` forces it to
the physical device. The cost of `fsync` (a disk round-trip, often 0.1–10 ms) is *the*
dominant durability cost in databases, which motivates:

- **Group commit (optimization, do later):** batch many pending records and `fsync` once
  for the whole batch. Amortizes the round-trip across N writers. This single technique is
  why real databases can do tens of thousands of durable writes/sec on one disk.
- **Durability knobs:** expose a policy — `fsync` every write (safe, slow) vs. every N ms
  (fast, may lose the last window on crash). Postgres (`synchronous_commit`) and Redis
  (`appendfsync`) both expose exactly this trade-off. Make it a config field.

> **Atomicity note:** an `UPDATE` is logically delete+insert. Logging it as **one** record
> (type 3) makes it atomic for free — replay either sees the whole update or, if it was the
> torn tail record, none of it. Never split an atomic operation across two records.

### 2.3 Snapshots / checkpoints

Replaying a WAL that's grown for weeks is slow and the file grows unbounded. Fix both with
periodic checkpoints.

**Snapshot file format** — serialize enough to rebuild the index without the graph search:
```
header     : magic, version, dim, metric, lsn_at_snapshot:u64, node_count:u64
ID maps    : next_ext_id, then node_count × (ext_id:u64, deleted:u8)
per node   : neighbours: for each layer, (count:u16, InternalId[count])
             vector data: f32[dim]
graph meta : entry_point:InternalId, max_layer:i32
```

**Checkpoint procedure:**
1. Write a **new** snapshot file (`snapshot.NNNN.tmp`), `fsync` it, then atomically
   `rename()` to `snapshot.NNNN` (rename is atomic on POSIX — a half-written snapshot
   never replaces a good one).
2. Record `lsn_at_snapshot` = the LSN of the last record included.
3. Append a `CHECKPOINT` record to the WAL.
4. Now safe to delete WAL segments / truncate the WAL up to that LSN.

**Fundamental:** the **snapshot + WAL tail** pair is universal — Redis (RDB + AOF),
Postgres (base backup/checkpoint + WAL), etcd/Raft (snapshot + log). Learn it once here.

### 2.4 Recovery (startup)

This routine is the *payoff* — it's what makes `kill -9` mid-write safe.

```
1. Find newest valid snapshot.NNNN; load it → in-memory index + ID maps.
   Set current_lsn = snapshot.lsn_at_snapshot. (No snapshot? start empty, lsn=0.)
2. Open the WAL. Validate the file header (magic/version/dim/metric must match).
3. Scan records from the start:
     - skip records with lsn <= current_lsn (already in the snapshot).
     - for each newer record: verify crc32c.
         * good  → apply to in-memory state, advance current_lsn.
         * bad   → STOP. This is the torn tail. Truncate the WAL here so future
                   appends start from a clean boundary.
4. Open the WAL for appending; next write uses lsn = current_lsn + 1.
```

Key insight: **idempotency / monotonic LSN** is what makes replay safe to run after *any*
crash, including a crash during recovery. You can always re-run it and converge.

### 2.5 Implementation plan for Phase 2

- New module `include/wal.h` / `src/wal.cpp`: `class Wal` with
  `open()`, `append(const Record&)`, `sync()`, `replay(callback)`, `truncate(lsn)`.
- New module `snapshot.h/.cpp`: `save(index, path)` / `load(path) -> index`.
- A `Storage` / `DurableVDB` layer that owns `VDB` + `Wal` and enforces the
  log-before-apply ordering. The pure `HNSWIndex` stays persistence-agnostic and testable.
- `VDBConfig` gains: `std::string data_dir`, `DurabilityPolicy {Always, EveryMs, Never}`,
  `checkpoint_threshold` (e.g. WAL bytes or op count).
- Tests:
  - round-trip: insert N, destroy object, reopen from disk, all N searchable.
  - **crash injection:** write a torn record (truncate the WAL mid-record), assert recovery
    drops exactly the torn tail and keeps everything before it.
  - snapshot + replay: checkpoint, add more ops, reopen, verify combined state.

---

## Phase 3 — Concurrency (Isolation)

**Fundamental:** isolation is the hardest ACID property. The progression below is itself
the lesson — each step teaches why the next exists.

### 3.1 Single global `shared_mutex`
Many concurrent readers **or** one writer. Correct and trivial; `search` takes a shared
lock, mutations take an exclusive lock. Bottleneck: writes block all reads.

### 3.2 Finer-grained locking
A per-node lock (or lock striping) taken during graph link edits, so inserts in different
regions of the graph proceed in parallel. This is roughly what `hnswlib` does. Teaches
lock ordering and deadlock avoidance.

### 3.3 (Stretch) Lock-free reads via versioning / RCU
Readers traverse an immutable snapshot of neighbor lists while writers publish new versions.
Introduces the idea behind **MVCC** — readers never block writers. Optional, advanced.

**Plan:** start at 3.1 (small, correct), add a concurrency stress test (N reader threads +
M writer threads, assert no crashes / no lost inserts), then iterate to 3.2 with the test
as your safety net.

---

## Phase 4 — Richer Queries

### 4.1 Metadata / payload store
A per-`ExternalId` typed blob or JSON. Persist it via the **same WAL** (extend records with
optional payload) so metadata is as durable as vectors.

### 4.2 Filtered search
"Nearest neighbors **where** category = X." Genuinely hard with ANN graphs. Three strategies
to study and benchmark:
- **Post-filter:** search K′ > K, drop non-matches. Simple; fails when the filter is selective.
- **Pre-filter:** restrict the candidate set up front (needs an attribute index).
- **Filtered traversal:** check the predicate during graph search. Best recall, most work.

**Plan:** implement post-filter first (cheap), add an attribute index, then filtered
traversal; benchmark recall vs. filter selectivity using `bench/`.

---

## Phase 5 — Interface & Operability

- **Library API hardening:** error types, input validation (dim mismatch, NaNs), RAII over
  raw `const float*`.
- **Network service (optional):** gRPC or HTTP server exposing insert/search/delete so VDB
  is a *service*, not just a linked library. Teaches request framing, backpressure, batching.
- **Config & schema persistence:** dim + metric saved with the data (the WAL/snapshot headers
  already carry this — enforce it on open).
- **Observability:** structured logging, metrics (QPS, recall, p99 latency, WAL fsync time,
  checkpoint duration), and growing out `bench/main.cpp` + `bench/mem.cpp`.

---

## Suggested first three PRs

1. **Phase 1.1 + 1.2:** external IDs + tombstone deletes (foundation for everything).
2. **Phase 2.1–2.2:** WAL file format + log-before-apply write path + replay-on-open
   (no snapshots yet — prove durability end to end first).
3. **Phase 2.3–2.4:** snapshots + checkpoint + WAL truncation, with the crash-injection test.

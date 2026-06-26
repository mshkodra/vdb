# VDB — A Vector Database From Scratch: Roadmap & Study Guide

> **Purpose of this document.** This is the spine of a learn-by-building project. The
> goal is *not* to produce a vector database as fast as possible — it is to deeply
> understand every layer of one, well enough to **write a technical analysis of the
> outcomes** at the end (see [The Final Write-Up](#the-final-write-up)). Each stage is
> structured the same way:
>
> 1. **What you'll learn** — the concepts this stage teaches.
> 2. **Theory / background** — the ideas, in enough depth to make decisions.
> 3. **Design decisions you own** — the forks in the road. *Decide these yourself
>    before writing code.* This is where the real learning is.
> 4. **Implementation tasks** — concrete, checkable steps.
> 5. **How to validate** — how you know it's correct.
> 6. **Resources** — papers, courses, and articles, best-first.
> 7. **Capture for the write-up** — the measurements/observations to record now so
>    the final analysis isn't reconstructed from memory.
>
> **How to work through it.** For each stage: read the theory and resources first,
> write down your answers to the design decisions in a lab notebook (`docs/notebook.md`
> — gitignored), *then* implement. Resist letting an assistant make the decisions; use
> it to check your understanding and review code after you've drafted the approach.

---

## Table of Contents

- [North Star: what "done" means](#north-star-what-done-means)
- [Background primer: the vector search problem](#background-primer-the-vector-search-problem)
- [Repository layout](#repository-layout)
- [Stage 0 — Foundations, tooling & the evaluation harness](#stage-0--foundations-tooling--the-evaluation-harness)
- [Stage 1 — Distance metrics & the brute-force baseline](#stage-1--distance-metrics--the-brute-force-baseline)
- [Stage 2 — IVF: clustering-based ANN](#stage-2--ivf-clustering-based-ann)
- [Stage 3 — HNSW: graph-based ANN](#stage-3--hnsw-graph-based-ann)
- [Stage 4 — Comparative benchmarking & analysis](#stage-4--comparative-benchmarking--analysis)
- [Stage 5 — The database layer: identity, deletes, metadata](#stage-5--the-database-layer-identity-deletes-metadata)
- [Stage 6 — Durability: WAL, snapshots, recovery](#stage-6--durability-wal-snapshots-recovery)
- [Stage 7 — Concurrency & isolation](#stage-7--concurrency--isolation)
- [Stage 8 — Interface & operability (optional)](#stage-8--interface--operability-optional)
- [The Final Write-Up](#the-final-write-up)
- [Master resource list](#master-resource-list)

---

## North Star: what "done" means

By the end you will have:

- Three interchangeable indexes behind one interface — **brute force** (exact),
  **IVF** (clustering), **HNSW** (graph) — and a clear, measured understanding of
  when each wins.
- A **database layer** on top: stable external IDs, deletes, metadata, durability
  (crash-safe via a WAL + snapshots), and thread-safe concurrent access.
- A **reproducible benchmark harness** that produces recall-vs-throughput curves.
- A **written analysis** that could stand as a portfolio piece or a talk: it explains
  the algorithms, justifies the design decisions, and interprets the measurements.

You are explicitly *not* trying to beat FAISS. You are trying to be able to explain,
from first principles, every number your system produces.

> A note on accuracy of resources: links rot and arXiv IDs are stable. Where a link
> is uncertain, search the **title + authors** given. Every resource below is real and
> findable by title.

---

## Background primer: the vector search problem

**Embeddings.** Modern ML maps things (text, images, audio) to fixed-length vectors
("embeddings") where geometric closeness ≈ semantic similarity. A vector database
stores millions–billions of these and answers: *given a query vector, return the K
most similar stored vectors.* This is **k-nearest-neighbor (kNN) search**.

**Why it's hard — the curse of dimensionality.** In low dimensions you'd use a k-d
tree or grid. In the 128–1536 dimensions typical of embeddings, space-partitioning
trees degrade to scanning nearly everything, and distances "concentrate" (the nearest
and farthest points become almost equidistant). Exact kNN in high dimensions has no
known sublinear algorithm that beats brute force in the worst case.

**The escape hatch — Approximate NN (ANN).** Give up *exactness*. Return neighbors
that are *almost always* the true nearest, much faster. This introduces the single
most important concept in the whole project:

> **The recall/throughput trade-off.** Every ANN method has a knob (IVF's `nprobe`,
> HNSW's `ef`) that trades **recall** (fraction of true neighbors found) against
> **throughput** (queries/sec). There is no single "score" — there is a *curve*.
> Comparing methods means comparing curves, not points.

**Two families you'll build:**
- **Partition-based (IVF):** cluster the data; search only nearby clusters. Simple,
  memory-light, great with quantization at billion scale.
- **Graph-based (HNSW):** build a navigable proximity graph; walk it greedily. Usually
  the best recall/latency on a single machine for moderate scale, at higher memory cost.

**Key metrics you will measure repeatedly:**
- **Recall@K** = |returned ∩ true top-K| / K, averaged over queries.
- **Throughput (QPS)** and **latency** (mean, p50, p95, p99).
- **Build time** and **memory footprint**.
- **Index parameters** that produced them (so results are reproducible).

Resources for this primer:
- Pinecone, *"What is a Vector Database?"* and their HNSW/IVF explainer articles —
  best gentle intuition.
- *ANN-Benchmarks* (Aumüller, Bernhardsson, Faithfull), website **ann-benchmarks.com** —
  see how the field formally compares methods (recall vs QPS Pareto fronts).
- Indyk & Motwani (1998), *"Approximate Nearest Neighbors: Towards Removing the Curse
  of Dimensionality"* — the foundational ANN/LSH paper (background, not required).

---

## Repository layout

```
include/
  vdb_types.h     Common ids/typedefs (InternalId, ExternalId, DistanceFn)
  distance.h      Distance metrics + Metric enum                  [Stage 1]
  index.h         Abstract Index interface (train/add/search)     [Stage 0/1]
  brute_index.h   Exact linear-scan baseline (ground truth)       [Stage 1]
  ivf_index.h     Inverted-file / clustering index                [Stage 2]
  hnsw_index.h    Hierarchical navigable small-world graph        [Stage 3]
  vdb.h           Top-level DB: ids, deletes, durability owner     [Stage 5+]
src/              Implementations (all stubbed with TODOs today)
tests/            Minimal harness (test.h) + per-stage test files
bench/            Comparative benchmark harness                    [Stage 4]
Makefile          `make test`, `make bench`, `make`
```

The `Index` interface already encodes one design decision (see Stage 0). Everything
in `src/` is a stub returning defaults — the build is intentionally green so you can
fill in one method at a time and watch tests go from red to green.

---

## Stage 0 — Foundations, tooling & the evaluation harness

**What you'll learn:** how to *measure* before you optimize; why an evaluation
harness is the most important code in an ANN project; interface design for
pluggable algorithms.

**Theory / background.** You cannot improve what you cannot measure, and ANN work is
*entirely* about measurement (recall vs speed). Build the ruler before the thing being
measured. The ruler has three parts: (1) **data generators**, (2) a **ground-truth**
computation (exact kNN), and (3) **metrics** (recall@K, QPS, latency percentiles).

Synthetic data matters: **uniform-random** vectors are the *worst case* for ANN
(distances concentrate, structure is absent), while **clustered** (mixture-of-Gaussians)
data resembles real embeddings, which live on lower-dimensional manifolds. Measuring on
both is itself a finding worth writing up.

**Design decisions you own:**
- **The `Index` interface.** Look at `include/index.h`. Is `train()` on the base class
  the right call (IVF needs it, HNSW/brute don't)? Should `search` return distances or
  just ids? Should there be a batched `add`? Decide and justify in your notebook.
- **Metric handling.** Is distance a `std::function` (flexible, not inlined) or a
  template parameter (fast, rigid)? You chose `std::function` in `vdb_types.h` — note
  the performance trade-off so you can revisit it in Stage 4.
- **Data model for ground truth:** precompute and cache exact neighbors per query set,
  or recompute each run? At what N does recompute become too slow?

**Implementation tasks:**
1. A data module (in `bench/`): `random_vectors(n, dim, seed)` and
   `clustered_vectors(n, dim, k_clusters, seed)`. Seed everything for reproducibility.
2. A ground-truth function: exact top-K via the brute index (Stage 1) or a local helper.
3. A metrics module: `recall_at_k`, plus a timing harness that reports mean/p50/p95/p99
   latency and QPS. Percentiles require sorting per-query latencies — don't just average.
4. A results table printer (CSV-friendly so you can plot later).

**How to validate:** recall of brute-force-vs-itself must be 100%. Latency numbers
should be stable across runs (fixed seeds, warm-up iterations discarded).

**Resources:**
- *ANN-Benchmarks* paper & repo (Aumüller et al.) — copy their methodology: fixed query
  sets, recall-vs-QPS curves, multiple parameter settings per method.
- Brendan Gregg, *Systems Performance* — for latency percentiles and why averages lie.
- Google Benchmark / nanobench (libraries) — optional, for rigorous microbenchmarking.

**Capture for the write-up:** your benchmarking methodology (data, seeds, hardware,
how recall and latency are defined). This becomes the "Experimental Setup" section.

---

## Stage 1 — Distance metrics & the brute-force baseline

**What you'll learn:** the metrics that define "similar"; why the exact baseline is
indispensable; first contact with SIMD/cache effects.

**Theory / background.** Three metrics cover almost everything:
- **Squared Euclidean (L2²)**: `Σ(aᵢ−bᵢ)²`. Monotonic with L2 distance but skips the
  `sqrt` — for ranking you never need the sqrt. Good default.
- **Inner product**: `Σ aᵢbᵢ`. Used by many models (e.g. dot-product retrieval). Note
  it's a *similarity* (bigger = closer), so store it negated to keep "smaller = closer"
  uniform across the codebase — that's why `distance.h` calls it `neg_inner_product`.
- **Cosine**: inner product of L2-normalized vectors = `1 − a·b/(‖a‖‖b‖)`. If you
  normalize vectors at insert time, cosine *reduces to* inner product — a nice
  optimization and a good thing to demonstrate empirically.

The **brute-force index** scans all vectors and partial-sorts the top K. It's `O(N·dim)`
per query — slow, but **exactly correct**, so it is the oracle for every recall number
you will ever report. It's also the honest baseline: if an ANN method isn't meaningfully
faster than brute force at your scale, that's a finding.

**Design decisions you own:**
- **Normalization policy** for cosine: normalize-on-insert (fast queries, mutates data)
  vs normalize-on-the-fly (keeps raw data). Pick one; note the trade-off.
- **Storage layout:** `vector<vector<float>>` (simple, pointer-chasing, poor locality)
  vs a single flat `vector<float>` of `N*dim` (contiguous, cache-friendly). This choice
  measurably affects brute-force speed — a great thing to benchmark in Stage 4.
- **Top-K selection:** full sort vs `std::partial_sort` vs a bounded max-heap. Different
  complexity (`N log N` vs `N log K`); measure which wins at your K.

**Implementation tasks:**
1. Implement the three functions in `src/distance.cpp`.
2. Implement `BruteIndex` in `src/brute_index.cpp` (`add`, then `search` with partial_sort).
3. Tests (`tests/test_brute.cpp`): tiny hand-checked cases where you know the answer;
   a randomized test asserting `search` returns vectors in nondecreasing distance order.

**How to validate:** hand-computed distances on 3–4 vectors; the returned order is sorted;
querying with a stored vector returns itself at distance 0 (for L2).

**Resources:**
- Intel *Intrinsics Guide* (software.intel.com) and a short SSE/AVX tutorial — for a
  later SIMD L2 kernel. Don't optimize yet; just know it's coming.
- FAISS source `faiss/utils/distances.cpp` — see how a production library writes these
  (blocked, SIMD, with an FMA-friendly `‖a‖² + ‖b‖² − 2a·b` expansion for L2).
- Agner Fog's optimization manuals — for why memory layout dominates at this scale.

**Capture for the write-up:** brute-force QPS vs N and vs dim (your baseline curve);
the effect of storage layout if you test both. This anchors every later speedup claim.

---

## Stage 2 — IVF: clustering-based ANN

**What you'll learn:** unsupervised clustering (k-means), Voronoi partitioning, the
coarse-quantizer idea, and your first real recall/speed knob (`nprobe`).

**Theory / background.** IVF = **Inverted File** index.

1. **Train.** Run **k-means** on a sample of the data to find `nlist` **centroids**.
   These centroids partition space into `nlist` **Voronoi cells** (each point belongs to
   the cell of its nearest centroid). The centroids are a "coarse quantizer."
2. **Add.** For each vector, find its nearest centroid and append its id to that cell's
   **inverted list** (cell → list of member vector ids).
3. **Search.** Find the `nprobe` centroids nearest the query, then brute-force only the
   vectors in those `nprobe` lists. You've replaced "scan N" with "scan N·(nprobe/nlist)".

The knobs:
- **`nlist`** (number of cells): more cells = smaller lists = faster, but training cost
  and the risk that the true neighbor sits in an unprobed cell both rise. A classic
  heuristic is `nlist ≈ √N`.
- **`nprobe`** (cells searched per query): the **recall/speed knob**. `nprobe = nlist`
  degenerates to brute force (recall 100%, no speedup). `nprobe = 1` is fastest, lowest
  recall. Sweeping `nprobe` traces IVF's recall-vs-QPS curve.

**k-means itself** (Lloyd's algorithm): assign each point to nearest centroid → recompute
each centroid as the mean of its members → repeat until stable. Initialization matters a
lot; **k-means++** seeding gives far better, more stable clusters than random seeding.
Watch for **empty clusters** (a centroid no point claims) — you must handle them (e.g.
re-seed from the largest cluster).

**Why IVF matters at scale:** real systems pair IVF with **Product Quantization (PQ)** to
compress vectors ~8–64×, enabling billion-scale search in RAM. PQ is an excellent
*stretch goal* for this stage and a strong write-up section, but get plain IVF correct
first.

**Design decisions you own:**
- **`nlist` and the training sample size.** Train on all data or a subset? (FAISS trains
  on a sample.) How does sample size affect centroid quality?
- **k-means initialization:** random vs k-means++. Implement at least random; ideally
  compare both — the difference in recall is a great empirical result.
- **Centroid distance during search:** exact scan of all `nlist` centroids (fine when
  `nlist` is small) vs something fancier. Note where this becomes the bottleneck.
- **Stretch:** add **IVF-PQ**. Decide the number of subquantizers `m` and bits per code.

**Implementation tasks:**
1. k-means in `src/ivf_index.cpp::train` (Lloyd's; handle empty clusters; fixed seed).
2. `add`: nearest-centroid assignment → push to inverted list; also store the raw vector.
3. `search`: rank centroids, gather candidates from `nprobe` lists, top-K over candidates.
4. Tests (`tests/test_ivf.cpp`): on clustered data with well-separated clusters,
   `nprobe=1` should already give high recall; recall must be **monotonically
   non-decreasing in `nprobe`**, reaching ~100% at `nprobe=nlist`.

**How to validate:** the monotonicity check above is the key invariant. Also: every added
vector lands in exactly one inverted list; total list lengths == size().

**Resources:**
- Jégou, Douze, Schmid (2011), *"Product Quantization for Nearest Neighbor Search"*,
  IEEE TPAMI — the IVF+PQ foundation. Read §II–III for the IVF idea even if you skip PQ.
- Arthur & Vassilvitskii (2007), *"k-means++: The Advantages of Careful Seeding"* — the
  initialization that makes k-means reliable.
- FAISS wiki (github.com/facebookresearch/faiss/wiki), *"The index factory"* and
  *"Guidelines to choose an index"* — practical `nlist`/`nprobe`/PQ choices.
- Pinecone, *"Product Quantization"* and *"IVF"* explainers — intuitive diagrams.

**Capture for the write-up:** IVF recall-vs-QPS as you sweep `nprobe`, for several
`nlist`; build (training) time vs `nlist`; (if done) the recall/memory effect of PQ.
A plot of the Voronoi cells in 2D makes a great figure.

---

## Stage 3 — HNSW: graph-based ANN

**What you'll learn:** small-world graphs, greedy search, the skip-list analogy, the
heuristic neighbor selection that makes HNSW work, and the `M`/`ef` knobs.

**Theory / background.** HNSW = **Hierarchical Navigable Small World** graph.

Build intuition in three layers of ideas:
1. **Greedy search on a proximity graph (NSW).** Connect each point to some near
   neighbors. To search, start anywhere and repeatedly hop to the neighbor closest to the
   query until no neighbor is closer. On a graph with both short links (accuracy) and a
   few long links ("small-world" shortcuts), this finds near neighbors in ~log(N) hops.
2. **Hierarchy (the skip-list analogy).** A skip list speeds linked-list search with
   sparse express lanes on top. HNSW does the same for the graph: **upper layers are
   sparse** (long hops, coarse approach), **lower layers dense** (fine search). A node's
   top layer is sampled from an exponential distribution (factor `mL = 1/ln(M)`), so few
   nodes reach high layers. Search descends layer by layer from one entry point.
3. **The neighbor-selection heuristic.** Naively linking the `M` closest candidates
   clusters all links in one direction and hurts connectivity. HNSW's heuristic
   (paper Algorithm 4) keeps a candidate only if it's closer to the new node than to any
   already-selected neighbor — spreading links across *directions*, which is what makes
   the graph navigable. Getting this right is the difference between great and mediocre
   recall.

The knobs:
- **`M`**: neighbors per node per layer. Higher `M` = better recall + more memory + slower
  build. Layer 0 typically allows `Mmax0 = 2M`.
- **`efConstruction`**: candidate-list width during build. Higher = better graph, slower
  build.
- **`ef` (search)**: the **recall/speed knob** at query time. Higher `ef` = explore more
  = higher recall, lower QPS. `ef ≥ K` always.

The core building block is **`search_layer(query, entry_points, ef, layer)`**: a greedy
best-first expansion using a candidate min-heap and a result max-heap of size `ef`,
tracking visited nodes. You use it with `ef=1` for the coarse descent through upper
layers and with the full `ef` at layer 0.

**Design decisions you own:**
- **Static vs incremental build.** Insert-one-at-a-time (what the interface implies, and
  what enables online updates) vs bulk build. Incremental is the interesting, database-y
  choice — keep it.
- **Entry-point & max-layer maintenance** as nodes are inserted (when a new node's
  sampled layer exceeds the current max, it becomes the new entry point).
- **Neighbor pruning on the *neighbor's* side.** When you link new↔existing and the
  existing node now exceeds `Mmax`, you must re-run selection to prune *its* list. Decide
  how (re-run the heuristic vs keep-closest). This is easy to get subtly wrong.
- **Determinism:** the level sampler uses RNG. Seed it for reproducible benchmarks.

**Implementation tasks:**
1. `sample_layer()` (exponential via `floor(-ln(U)·mL)`).
2. `search_layer()` (the heap-based greedy expansion).
3. `select_neighbors()` (the directional heuristic; consider keepPrunedConnections).
4. `add()` (sample layer; greedy-descend with ef=1 above the node's layer; at each layer
   ≤ its top, search with efConstruction, select neighbors, link both directions, prune
   over-full neighbors; update entry point/max layer).
5. `search()` (descend ef=1 to layer 0, then ef-search, return top-K).
6. Tests (`tests/test_hnsw.cpp`): recall vs brute force ≥ ~0.95 at reasonable `ef` on
   clustered data; recall **monotonically non-decreasing in `ef`**; self-query returns
   self; graph stays connected (every node reachable from the entry point at layer 0).

**How to validate:** the monotonic-in-`ef` invariant; reachability of all non-deleted
nodes; recall approaching 1.0 as `ef → N`.

**Resources (this is the one to read carefully):**
- **Malkov & Yashunin (2018/2016), *"Efficient and Robust Approximate Nearest Neighbor
  Search Using Hierarchical Navigable Small World Graphs"*, IEEE TPAMI — arXiv:1603.09320.**
  The primary source. Algorithms 1–5 are exactly what you implement. Read it twice.
- Malkov et al. (2014), *"Approximate nearest neighbor algorithm based on navigable small
  world graphs"*, Information Systems — the NSW predecessor; clarifies the greedy-search idea.
- **hnswlib** (github.com/nmslib/hnswlib) — a compact, readable reference implementation.
  Use it to check your understanding *after* drafting, not to copy.
- Pinecone, *"Hierarchical Navigable Small Worlds (HNSW)"* — the best visual explainer.
- Kleppmann's skip-list explanation (or any) — to ground the hierarchy analogy.

**Capture for the write-up:** HNSW recall-vs-QPS sweeping `ef`; sensitivity to `M` and
`efConstruction`; build time vs `M`; memory (bytes/vector ≈ vector size + `~M` ids/layer).
A side-by-side of the *same* recall achieved by HNSW vs IVF, at their respective QPS, is
the headline result of the project.

---

## Stage 4 — Comparative benchmarking & analysis

**What you'll learn:** how to run a fair, reproducible comparison and turn raw numbers
into defensible conclusions — the analytical core of the final write-up.

**Theory / background.** A single (recall, QPS) point is meaningless in isolation; the
**Pareto frontier** (recall vs QPS as each method's knob is swept) is the unit of
comparison. A method "wins" in a region if its curve is up-and-to-the-right of the
others there. Often the answer is "it depends on the target recall and scale" — and
showing *where* each method wins is exactly the insight you want.

Confounders to control: same data, same query set, same ground truth, warm caches,
discarded warm-up runs, single-threaded for apples-to-apples (until Stage 7), and
*reported parameters* for every point.

**Design decisions you own:**
- **The grid:** which `(N, dim, data-distribution)` settings, and which knob values per
  method. Enough points to draw a smooth curve; few enough to run in minutes.
- **What "fair" means:** do you cap build time? Memory? Report all three (recall, QPS,
  memory, build) so readers can pick their own constraint.
- **Output format:** emit CSV; plot with a small Python/gnuplot script (in `docs/`).

**Implementation tasks:**
1. Flesh out `bench/main.cpp`: for each method × parameter setting, measure recall@K,
   QPS, latency percentiles, build time, and memory; print CSV.
2. A plotting script producing: (a) recall-vs-QPS per method, (b) build-time vs N,
   (c) memory vs N, (d) recall-vs-QPS on uniform vs clustered data.
3. A short results notes file capturing what you observe.

**How to validate:** brute force sits at recall 1.0 by construction; IVF and HNSW curves
are monotonic; re-running yields stable numbers (seeded). Sanity-check against published
intuition (HNSW usually dominates IVF on recall/latency at this scale; IVF wins on
memory/build-time and at very large scale with PQ).

**Resources:**
- *ANN-Benchmarks* (ann-benchmarks.com) — the gold standard for *how* to present this;
  mirror their plot style.
- Aumüller, Bernhardsson, Faithfull (2020), *"ANN-Benchmarks: A benchmarking tool for
  approximate nearest neighbor algorithms,"* Information Systems.
- *"Billion-scale similarity search with GPUs"* (Johnson, Douze, Jégou, arXiv:1702.08734)
  — for scale context and the IVF-PQ regime.

**Capture for the write-up:** *everything here is the write-up's Results section.* Save
the CSVs and the plots; write a paragraph interpreting each figure.

---

## Stage 5 — The database layer: identity, deletes, metadata

**What you'll learn:** the indirection that separates a *database* from an *index*;
tombstone deletes and compaction; the ACID "consistency" of invariants.

**Theory / background.** Indexes speak `InternalId` (an array offset — unstable across
compaction). A database must give callers a **stable `ExternalId`** and translate at the
boundary. That indirection (`ext→int` and `int→ext` maps) is what lets you delete, move,
and compact data underneath callers without breaking their references — the same idea as
a filesystem inode or a DB primary key vs row location.

**Deletes** can't cleanly remove an HNSW node (its links are load-bearing) or shrink an
IVF list mid-flight cheaply. The universal answer: **tombstones** — mark deleted, exclude
from results, *reclaim space later* via **compaction** (rebuild from live nodes,
remapping internal ids; external ids unchanged). This "mark now, reclaim later" pattern
is exactly Postgres dead tuples + VACUUM and LSM-tree tombstones + compaction.

**Updates** = tombstone old + insert new under the *same* `ExternalId` (atomicity of the
pair matters once the WAL exists in Stage 6).

**Metadata/payload:** a per-`ExternalId` blob (JSON or typed) enabling later filtered
search ("nearest where category=X").

**Design decisions you own:**
- **Who owns the id maps & tombstones** — `VDB` (recommended; keeps indexes pure) vs each
  index. Owning it in `VDB` means deletes work uniformly across HNSW/IVF/brute.
- **Tombstone representation:** `vector<bool>` deleted flags vs a set of deleted ids.
- **Compaction trigger:** ratio of dead-to-live, absolute count, or manual. And: rebuild
  in place vs build-new-then-swap.
- **Filtered search strategy (if you add metadata):** post-filter (search K′>K, drop
  non-matches — fails when filter is selective), pre-filter (restrict candidates — needs
  an attribute index), or filtered traversal (check predicate during search). Just note
  the design now; implementing is optional.

**Implementation tasks:**
1. Add `ext_to_int_`, `int_to_ext_`, `next_ext_id_`, and tombstones to `VDB`.
2. `insert` returns an `ExternalId`; `search` translates ids back and skips tombstoned
   results (but the index still *traverses* through deleted HNSW nodes — keep them in the
   graph for connectivity).
3. `remove(ExternalId)`, `contains(ExternalId)`, `update(ExternalId, vec)`.
4. `compact()` rebuilding live data; assert external ids survive.
5. Tests (`tests/test_lifecycle.cpp`): delete-then-search excludes the vector; update
   changes results under a stable id; compaction preserves all live external ids.

**How to validate:** invariants — `ext_to_int_` and `int_to_ext_` are mutual inverses for
live ids; size() counts only live vectors; compaction is recall-neutral.

**Resources:**
- Kleppmann, *Designing Data-Intensive Applications*, Ch. 3 (storage engines: log
  structure, compaction, tombstones).
- CMU 15-445 (Andy Pavlo), lectures on storage & buffer management — 15445.courses.cs.cmu.edu.
- PostgreSQL docs on MVCC & VACUUM — the canonical "tombstone + reclaim" system.

**Capture for the write-up:** how deletes interact with each index type; compaction cost
vs reclaimed space; any recall change after many delete/insert cycles (graph degradation
is a real, writable phenomenon).

---

## Stage 6 — Durability: WAL, snapshots, recovery

> This is the deepest database stage. Read it slowly; it's worth its own write-up section.

**What you'll learn:** durability and atomicity (the "D" and "A" of ACID), write-ahead
logging, checkpoints, crash recovery, and the `fsync` discipline that underlies every
real database.

### 6.0 The mental model

Two mechanisms working together:
1. **Write-Ahead Log (WAL):** an append-only file. *Before* mutating in-memory state,
   append a record describing the change and flush it to disk. Crash → **replay** the log
   on startup to reconstruct state.
2. **Snapshot (checkpoint):** periodically serialize the whole index/DB to disk so
   recovery needn't replay from the beginning of time. After a snapshot, old WAL is
   truncated.

Recovery = **load newest snapshot, then replay the WAL records after it.**

The golden rule — and the literal meaning of *write-ahead*:

> **Log the intent durably BEFORE you change in-memory data.** If you mutate first and
> crash before logging, the write is lost but the caller was told it succeeded. Log
> first, and the worst case is a logged-but-unapplied record, which replay fixes.
> Ordering is everything.

### 6.1 On-disk WAL format

A self-describing, append-only, **checksummed** record stream.

**File header (once, at creation):**
```
magic      : u32   = 0x56444257   ("VDBW")
version    : u16   = 1
dim        : u32   // guards against opening a DB with mismatched config
metric     : u8    // 0=L2, 1=IP, 2=cosine
(pad to 32 bytes)
```

**Record framing (repeated):**
```
length     : u32   // bytes of [type..payload], excludes crc
crc32c     : u32   // checksum over [length..payload]
type       : u8    // INSERT=1, DELETE=2, UPDATE=3, CHECKPOINT=4
lsn        : u64   // Log Sequence Number, strictly increasing
payload    : depends on type
```
Payloads: `INSERT{ext_id:u64, vec:f32[dim]}`, `DELETE{ext_id:u64}`,
`UPDATE{ext_id:u64, vec:f32[dim]}`, `CHECKPOINT{snapshot_id:u64}`.

Why each field exists — the lessons:
- **length prefix** → frame/skip records without parsing them.
- **crc32c** → detect a **torn write** (a crash mid-`write()` leaving a half-record). On
  replay, a bad checksum means "this record and everything after is garbage; stop here."
  This is how you distinguish a clean tail from corruption. (CRC32C has a hardware
  instruction on modern x86/ARM — use it.)
- **lsn** → a monotonic name for every write; the vocabulary tying WAL to snapshots
  ("snapshot N contains all changes up to lsn=X") and making replay idempotent.
- **type** → makes the log a replayable operation journal, not just data.

### 6.2 The write path (the `fsync` discipline)

For each mutating op:
1. Serialize the record.
2. `write()` it to the WAL fd (lands in the OS page cache).
3. **`fsync()`/`fdatasync()` the WAL fd** — *this* is the durability point; it returns
   only once bytes are on stable storage.
4. *Only now* apply the change to in-memory index + id maps.
5. Return success to the caller.

**Why `fsync` is the whole game:** `write()` only copies into volatile page cache; a
power cut loses it. `fsync` forces the device. Its cost (a disk round-trip, ~0.1–10 ms)
is the dominant durability cost in databases, which motivates two things to build/measure:
- **Group commit:** batch many pending records and `fsync` once for the batch — amortizes
  the round-trip across writers. This single trick is why real DBs do tens of thousands of
  durable writes/sec on one disk.
- **Durability policy knob:** `fsync` every write (safe, slow) vs every N ms (fast, may
  lose the last window on crash). This is exactly Postgres `synchronous_commit` and Redis
  `appendfsync`. Make it config and *measure both points*.

> **Atomicity:** log an `UPDATE` as **one** record so replay sees all-or-nothing. Never
> split an atomic operation across two records.

### 6.3 Snapshots / checkpoints

Replaying a weeks-old WAL is slow and the file grows unbounded. Fix both:

**Snapshot file** serializes enough to rebuild without re-searching: header
(`magic, version, dim, metric, lsn_at_snapshot, node_count`), the id maps & tombstones,
per-node neighbours (HNSW) or inverted lists + centroids (IVF), the raw vectors, and graph
meta (entry point, max layer).

**Checkpoint procedure (crash-safe via atomic rename):**
1. Write `snapshot.NNNN.tmp`; `fsync` it; `rename()` → `snapshot.NNNN`. POSIX `rename`
   is atomic — a half-written snapshot never replaces a good one.
2. Record `lsn_at_snapshot` = last included lsn.
3. Append a `CHECKPOINT` record to the WAL.
4. Now safe to truncate/delete WAL up to that lsn.

The **snapshot + WAL-tail** pair is universal: Redis (RDB+AOF), Postgres
(checkpoint+WAL), etcd/Raft (snapshot+log). Learn it once here.

### 6.4 Recovery (startup) — the payoff

```
1. Find newest valid snapshot.NNNN; load it. current_lsn = its lsn_at_snapshot.
   (No snapshot → start empty, current_lsn = 0.)
2. Open WAL; validate header (magic/version/dim/metric must match — refuse otherwise).
3. Scan records from the start:
     - skip records with lsn <= current_lsn (already in snapshot).
     - for each newer record, verify crc32c:
         good → apply to in-memory state, advance current_lsn.
         bad  → STOP (torn tail). Truncate the WAL here for a clean append boundary.
4. Reopen WAL for append; next write uses lsn = current_lsn + 1.
```
Key insight: **monotonic lsn + idempotent replay** make recovery safe to re-run after
*any* crash, including a crash *during* recovery. It always converges.

**Design decisions you own:**
- **Granularity of durability** (per-op vs group commit vs time-based) and the default.
- **Snapshot encoding** (hand-rolled binary vs a format like a length-prefixed scheme).
  Hand-rolled teaches the most; keep it simple and versioned.
- **What's logged for HNSW** — just the logical op (`INSERT ext_id, vec`) and rebuild the
  graph by replaying inserts, vs logging graph edges too. Logical ops are far simpler and
  smaller; the trade-off is replay does real insertion work. Discuss this — it's a genuine
  engineering fork (logical vs physical logging).
- **WAL segmentation** (one growing file vs rotating segments) — affects truncation.

**Implementation tasks:**
1. `crc32c` helper; a small serialize/deserialize helper for records.
2. `Wal` class: `open`, `append(record)`, `sync`, `replay(callback)`, `truncate(lsn)`.
3. `snapshot` module: `save(db, path)` / `load(path)`.
4. A `DurableVDB` (or fold into `VDB`) enforcing **log-before-apply**; config gains
   `data_dir`, `DurabilityPolicy`, `checkpoint_threshold`.
5. Tests (`tests/test_durability.cpp`):
   - round-trip: insert N, destroy object, reopen from disk, all N searchable.
   - **crash injection:** truncate the WAL mid-record; assert recovery drops exactly the
     torn tail and keeps everything before it.
   - snapshot+replay: checkpoint, add more ops, reopen, verify combined state.

**How to validate:** the crash-injection test is the real proof. Also: header mismatch is
rejected; recovery is idempotent (run it twice → same state).

**Resources:**
- Kleppmann, *DDIA*, Ch. 3 (log-structured storage) & Ch. 7 (transactions, atomicity,
  durability) — start here.
- Hellerstein, Stonebraker, Hamilton (2007), *"Architecture of a Database System"* —
  §3–4 on logging/recovery; the best single survey.
- Mohan et al. (1992), *"ARIES: A Transaction Recovery Method Supporting Fine-Granularity
  Locking and Partial Rollbacks Using Write-Ahead Logging,"* ACM TODS — the canonical WAL
  recovery paper (you're building a simplified ARIES).
- CMU 15-445 (Pavlo), the **Logging** and **Recovery (ARIES)** lectures — clear, modern.
- Redis *persistence* docs (RDB+AOF) and PostgreSQL *WAL/`synchronous_commit`* docs —
  real systems exposing exactly your knobs.
- *"Files are hard"* (Dan Luu) and the *fsyncgate* discussion — sobering reading on how
  `fsync`/error handling actually behaves on real filesystems.

**Capture for the write-up:** durable-writes/sec under per-op `fsync` vs group commit vs
no-sync (a dramatic chart); recovery time vs WAL length and vs snapshot frequency; the
crash-injection result as evidence of correctness. This is a standout write-up section.

---

## Stage 7 — Concurrency & isolation

**What you'll learn:** the "I" of ACID; reader/writer synchronization; why coarse locks
bottleneck and how real ANN libs go finer; the road toward lock-free/MVCC.

**Theory / background.** Isolation is the hardest ACID property. Progress in steps, each
teaching why the next exists:
1. **Single global `std::shared_mutex`:** many readers *or* one writer. `search` takes a
   shared lock; mutations take an exclusive lock. Correct and trivial. Bottleneck: writes
   block all reads.
2. **Finer-grained locking:** per-node (or striped) locks taken only while editing graph
   links, so inserts in different graph regions proceed in parallel — roughly what hnswlib
   does. Teaches lock ordering and deadlock avoidance.
3. **(Stretch) Lock-free reads via versioning / RCU:** readers traverse an immutable
   snapshot of neighbor lists while writers publish new versions — the idea behind **MVCC**
   (readers never block writers).

**Design decisions you own:**
- **Concurrency model:** single-writer/multi-reader (simplest, often enough) vs
  multi-writer. Most embedded vector stores are single-writer — a defensible, simpler choice.
- **Where the lock lives:** in `VDB` (coarse) vs inside each index (fine). Interaction with
  the WAL: the `fsync` must not hold the index write lock longer than necessary (or
  throughput craters) — a subtle, writable interaction between Stages 6 and 7.

**Implementation tasks:**
1. Wrap mutations/reads with a `shared_mutex` in `VDB`; make the WAL append thread-safe.
2. A concurrency stress test (`tests/test_concurrency.cpp`): N reader threads + M writer
   threads; assert no crashes, no lost inserts, no torn reads; run under TSan/ASan.
3. (Optional) move to per-node locks in HNSW; keep the stress test as the safety net.

**How to validate:** clean runs under `-fsanitize=thread` and `-fsanitize=address`; insert
count under concurrency equals expected; recall unchanged vs single-threaded.

**Resources:**
- CMU 15-445 (Pavlo), concurrency-control lectures; and the *latching vs locking*
  distinction (you're doing latching — protecting in-memory structures).
- C++ Concurrency in Action (Anthony Williams) — `shared_mutex`, memory model, RCU ideas.
- hnswlib source — how a real HNSW handles concurrent insert/search.
- Kleppmann, *DDIA*, Ch. 7 (isolation levels, MVCC) for the conceptual ladder.

**Capture for the write-up:** throughput scaling vs thread count (coarse vs fine locking);
the WAL/lock interaction; any recall/latency change under contention.

---

## Stage 8 — Interface & operability (optional)

**What you'll learn:** turning a library into a service; API hardening; observability.

- **Library API hardening:** typed errors, input validation (dim mismatch, NaNs/Infs),
  RAII spans instead of raw `const float*`.
- **Network service:** a small gRPC/HTTP server exposing insert/search/delete — teaches
  request framing, batching, backpressure.
- **Config/schema persistence:** dim + metric stored with the data (the WAL/snapshot
  headers already carry these — enforce on open).
- **Observability:** structured logs and metrics (QPS, recall, p99, WAL fsync time,
  checkpoint duration); grow `bench/` into a load generator.

**Resources:** gRPC C++ docs; Brendan Gregg on observability; the Prometheus data-model
docs for metric design.

**Capture for the write-up:** end-to-end latency including serialization; overhead of the
service layer vs the in-process library.

---

## The Final Write-Up

The deliverable that ties the project together. Aim for something you'd be proud to put in
a portfolio or present as a talk. A suggested structure:

1. **Abstract / TL;DR** — what you built and the top 3 findings.
2. **Problem & background** — vector search, the curse of dimensionality, the
   recall/throughput trade-off. (From the primer.)
3. **System design** — the layered architecture (index interface → IVF/HNSW/brute → DB
   layer → durability → concurrency); the major design decisions *and the alternatives you
   rejected and why* (this is what demonstrates understanding).
4. **Algorithms** — IVF and HNSW explained in your own words, with the key diagrams
   (Voronoi cells; the layered graph). Show you can derive *why* each knob trades recall
   for speed.
5. **Experimental setup** — data, query sets, hardware, metric definitions, seeds.
6. **Results** — the recall-vs-QPS curves; IVF vs HNSW vs brute; uniform vs clustered;
   build time and memory; durability throughput (fsync policies); recovery time;
   concurrency scaling. One paragraph of interpretation per figure.
7. **Findings & discussion** — where each method wins and *why*; surprises; the limits of
   your implementation vs FAISS-class systems.
8. **Lessons learned & future work** — PQ, filtered search, distributed sharding, etc.
9. **Reproducibility appendix** — exact commands, parameters, environment.

**Throughout the project, keep `docs/notebook.md`** (gitignored) — your design-decision
answers, dead ends, and raw numbers. The write-up is mostly assembled from it; don't try
to reconstruct it at the end.

The strongest possible write-up answers, with evidence: *"Given a target recall and a
constraint (latency, memory, or build time), which index should you choose, and why?"* —
and backs the durability/concurrency claims with the crash-injection and stress tests.

---

## Master resource list

**Courses**
- CMU 15-445/645 *Database Systems* (Andy Pavlo) — storage, logging, ARIES, concurrency.
  Free lectures at 15445.courses.cs.cmu.edu.
- MIT 6.824 *Distributed Systems* — if you pursue sharding/replication later.

**Books**
- Martin Kleppmann, *Designing Data-Intensive Applications* (2017).
- Hellerstein, Stonebraker, Hamilton, *Architecture of a Database System* (2007).
- Gray & Reuter, *Transaction Processing: Concepts and Techniques* (1992).
- Anthony Williams, *C++ Concurrency in Action* (2e).

**ANN / vector search papers**
- Malkov & Yashunin (2018), *HNSW*, IEEE TPAMI — arXiv:1603.09320. (Stage 3)
- Malkov et al. (2014), *Navigable Small World graphs*, Information Systems. (Stage 3)
- Jégou, Douze, Schmid (2011), *Product Quantization for NN Search*, IEEE TPAMI. (Stage 2)
- Arthur & Vassilvitskii (2007), *k-means++*. (Stage 2)
- Johnson, Douze, Jégou (2017), *Billion-scale similarity search with GPUs*,
  arXiv:1702.08734 (FAISS). (Stage 4)
- Indyk & Motwani (1998), *Approximate Nearest Neighbors / LSH*. (Background)
- Aumüller, Bernhardsson, Faithfull (2020), *ANN-Benchmarks*, Information Systems. (Stage 4)

**Database / durability**
- Mohan et al. (1992), *ARIES*, ACM TODS. (Stage 6)
- PostgreSQL docs: WAL, `synchronous_commit`, MVCC/VACUUM. (Stages 5–6)
- Redis docs: persistence (RDB + AOF). (Stage 6)
- Dan Luu, *"Files are hard"*; the Postgres *fsyncgate* writeups. (Stage 6)

**Reference implementations (read after drafting, not before)**
- hnswlib — github.com/nmslib/hnswlib (HNSW).
- FAISS — github.com/facebookresearch/faiss (IVF, PQ, distance kernels; see the wiki).

**Intuition / explainers**
- Pinecone learning center: HNSW, IVF, Product Quantization articles.
- Weaviate / Qdrant engineering blogs on HNSW internals and filtered search.

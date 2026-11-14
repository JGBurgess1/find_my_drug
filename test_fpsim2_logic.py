"""Unit-test the MPI-specific logic that cannot be exercised without a runtime."""
import importlib.util
import os
import sys
import tempfile

sys.argv = ["x"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


build = load("b", "fpsim2_build_db.py")
search = load("s", "fpsim2_similarity_search.py")

fails = []


def check(label, cond):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        fails.append(label)


# ── assign_shard: must partition the input exactly, for any rank count ───────
print("assign_shard partitioning")
tmp = tempfile.mkdtemp()
paths = []
for i in range(17):                       # deliberately prime-ish, uneven
    p = os.path.join(tmp, f"f{i}.smi")
    with open(p, "w") as fh:
        fh.write("C" * (i * 100 + 1))      # varying sizes
    paths.append(p)

for size in (1, 2, 3, 4, 5, 8, 17, 20):   # 20 > number of files
    assigned = [build.assign_shard(paths, r, size) for r in range(size)]
    flat = [p for part in assigned for p in part]
    check(f"size={size:>2}: covers every file exactly once",
          sorted(flat) == sorted(paths))
    counts = [len(a) for a in assigned]
    spread = max(counts) - min(counts)
    check(f"size={size:>2}: balanced within 1 file (spread={spread})",
          spread <= 1)

# largest-first dealing should spread big files across ranks, not pile them up
big = build.assign_shard(paths, 0, 4)
check("rank 0 gets the single largest file",
      max(paths, key=os.path.getsize) in big)

# ── search shard split: shards[rank::size] must also partition ──────────────
print("\nsearch shard split")
for n_shards in (1, 3, 7, 10):
    for size in (1, 2, 3, 4, 11):
        shards = list(range(n_shards))
        assigned = [shards[r::size] for r in range(size)]
        flat = [s for part in assigned for s in part]
        if sorted(flat) != shards:
            check(f"shards={n_shards} ranks={size}", False)
            break
    else:
        check(f"shards={n_shards}: partitions cleanly for all rank counts",
              True)

# ── merge_hits: cross-shard dedupe, ordering, top_n ─────────────────────────
print("\nmerge_hits")
rows = [
    ("q1", 100, "CCO", 0.90),
    ("q1", 100, "CCO", 0.95),   # same id from another shard, better coeff
    ("q1", 200, "CCC", 0.80),
    ("q1", 300, "CCN", 0.99),
    ("q2", 100, "CCO", 0.70),
]
merged = search.merge_hits(rows, top_n=0)
check("dedupes id across shards", len(merged["q1"]) == 3)
check("keeps the best coefficient for a duplicate",
      dict((i, c) for i, _s, c in merged["q1"])[100] == 0.95)
check("sorted by coefficient descending",
      [c for _i, _s, c in merged["q1"]] == [0.99, 0.95, 0.80])
check("queries kept separate", len(merged["q2"]) == 1)
check("top_n trims per query",
      len(search.merge_hits(rows, top_n=2)["q1"]) == 2)
check("top_n keeps the best ones",
      [i for i, _s, _c in search.merge_hits(rows, top_n=2)["q1"]] == [300, 100])

# ── init_mpi must degrade, not raise, with a broken mpi4py ──────────────────
print("\ninit_mpi fallback")
comm, rank, size = build.init_mpi()
check("build: falls back to serial without an MPI runtime",
      (comm, rank, size) == (None, 0, 1))
comm, rank, size = search.init_mpi()
check("search: falls back to serial without an MPI runtime",
      (comm, rank, size) == (None, 0, 1))
check("build: --no-mpi honoured", build.init_mpi(disable=True) == (None, 0, 1))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)

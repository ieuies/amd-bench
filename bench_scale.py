import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "lib"))
import seekdb_demo as sk

DIM, N, K, BATCH = 128, 50000, 10, 1000
rng = np.random.default_rng(42)
base = rng.random((N, DIM), dtype=np.float32)


def lit(row):
    return "'[" + ",".join("%.6f" % x for x in row) + "]'"


assert "not connected" not in str(sk.version()), "observer not running"
print("GPU:", sk.gpu_info()["name"], "| ROCm:", sk.gpu_info()["rocm"], flush=True)
print(f"synthetic base: {N} x {DIM}", flush=True)


def build_once(name, lib):
    sk.sql(f"use vec; drop table if exists {name}")
    sk.sql(f"use vec; create table {name}(c1 int primary key, c2 vector({DIM}), "
           f"vector index i(c2) with (distance=l2,type=hnsw,lib={lib},"
           f"m=16,ef_construction=200,ef_search=64))")
    t0 = time.perf_counter()
    for start in range(0, N, BATCH):
        vals = ",".join(f"({i},{lit(base[i])})" for i in range(start, min(start + BATCH, N)))
        sk.sql(f"use vec; insert into {name} values {vals}")
        if start % 10000 == 0:
            print(f"  {name} {start}/{N} {time.perf_counter() - t0:.1f}s", flush=True)
    probe = lit(base[0])
    last = None
    while time.perf_counter() - t0 < 1800:
        rows = sk.sql_rows(f"use vec; select c1 from {name} "
                           f"order by l2_distance(c2,{probe}) approximate limit {K}")
        cur = tuple(r[0] for r in rows[:K]) if rows else None
        if cur and cur == last:
            break
        last = cur
        time.sleep(0.5)
    return time.perf_counter() - t0


t_cpu = build_once("scale_cpu", "vsag")
print(f"CPU build {t_cpu:.2f}s", flush=True)
t_gpu = build_once("scale_gpu", "cuvs")
print(f"GPU build {t_gpu:.2f}s", flush=True)
print("")
print(f"| {N}/{DIM} | {t_cpu:.2f}s | {t_gpu:.2f}s | {t_cpu / t_gpu:.2f}x |")

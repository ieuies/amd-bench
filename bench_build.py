import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.getcwd(), "lib"))
import seekdb_demo as sk
DIM, ROUNDS, K = 128, 3, 10

def up():
    return "not connected" not in str(sk.version())

if not up():
    print("starting observer ...", flush=True)
    print(sk.start_observer("/workspace/run/obs1"))
    for _ in range(180):
        if up():
            break
        time.sleep(1)

ver = sk.version()
gi, si = sk.gpu_info(), sk.stack_info()
print("GPU:", gi["name"], "| arch:", gi["gfx"], "| ROCm:", gi["rocm"])
print("seekdb:", ver, "| bridge:", "OK" if si["bridge"] else "MISSING", "| libcuvs:", si["libcuvs_linked"])
assert up(), "observer still not ready"
sk.setup_data(10000)
print("t_vsag rows:", sk.sql_rows("use vec; select count(*) from t_vsag"))

def build_once(name, lib):
    sk.sql(f"use vec; drop table if exists {name}")
    sk.sql(f"use vec; create table {name}(c1 int primary key, c2 vector({DIM}), vector index i(c2) with (distance=l2,type=hnsw,lib={lib},m=16,ef_construction=200,ef_search=64))")
    t0 = time.perf_counter()
    sk.sql(f"use vec; insert into {name} select * from t_vsag")
    probe = sk.sql_rows("use vec; select pv from probes_100 where id=1")[0][0]
    last = None
    while time.perf_counter() - t0 < 300:
        rows = sk.sql_rows(f"use vec; select c1 from {name} order by l2_distance(c2,{probe}) approximate limit {K}")
        cur = tuple(r[0] for r in rows[:K]) if rows else None
        if cur and cur == last:
            break
        last = cur
        time.sleep(0.5)
    return time.perf_counter() - t0

cpu, gpu = [], []
for r in range(ROUNDS):
    cpu.append(build_once("cmp_cpu", "vsag")); print(f"round{r} cpu {cpu[-1]:.2f}s", flush=True)
    gpu.append(build_once("cmp_gpu", "cuvs")); print(f"round{r} gpu {gpu[-1]:.2f}s", flush=True)

def recall(name):
    probe = sk.sql_rows("use vec; select pv from probes_100 where id=1")[0][0]
    a = [r[0] for r in sk.sql_rows(f"use vec; select c1 from {name} order by l2_distance(c2,{probe}) approximate limit {K}")]
    e = [r[0] for r in sk.sql_rows(f"use vec; select c1 from {name} order by l2_distance(c2,{probe}) limit {K}")]
    return len(set(a) & set(e)) / max(1, len(e))

med = statistics.median
rc, rg = recall("cmp_cpu"), recall("cmp_gpu")
print("")
print("| data/dim | CPU build | GPU build | speedup | CPU recall@10 | GPU recall@10 |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
print(f"| 10000/{DIM} | {med(cpu):.2f}s | {med(gpu):.2f}s | {med(cpu)/med(gpu):.2f}x | {rc:.3f} | {rg:.3f} |")
print("raw cpu:", [round(x, 2) for x in cpu])
print("raw gpu:", [round(x, 2) for x in gpu])

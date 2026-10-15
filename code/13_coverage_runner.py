"""Runner for 13_coverage_sim.py: every task in its own subprocess (a native crash cannot take
down the run), cached (existing output files are skipped). Normal priority: with below-normal
priority the workers were starved by concurrent jobs (0.2 s CPU in 13 min).

Usage: python 13_coverage_runner.py <n_parallel> [deadline_minutes] [light]
('light': population + light sample tasks only). The design is taken from CSIM_DESIGN (wr|wor).
Task order: population fits, the R_TOTAL light sample tasks, then the N_VAR variance tasks
(r = 0, 1, ...). With a deadline, no new task is started after it; collect then uses every
completed variance task (the order r is random with respect to the data, so the subset is a
random subset of the samples).
"""
import importlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

here = Path(__file__).parent
sys.path.insert(0, str(here))
cs = importlib.import_module("13_coverage_sim")

n_par = int(sys.argv[1])
deadline = time.time() + 60 * float(sys.argv[2]) if len(sys.argv) > 2 else float("inf")
logdir = cs.OUT / "logs" / time.strftime("%Y%m%d_%H%M%S")
logdir.mkdir(parents=True, exist_ok=True)
C = cs.CNTRY
# BLAS single-threaded: with ~30 concurrent jobs on the machine, multithreaded BLAS in the OLS
# benchmark (96 threads per process) spin-waited and starved the workers; XGBoost uses nthread.
ENV = {**os.environ, "OMP_NUM_THREADS": os.environ.get("CSIM_NTHREAD", "1"), "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1"}
T = cs.TAG
tasks = [("--pop", None, cs.OUT / f"{C}_pop.pkl")]
tasks += [("--one", r, cs.OUT / f"{T}_s{r:03d}.pkl") for r in range(cs.R_TOTAL)]
if "light" not in sys.argv[3:]:
    tasks += [("--var", r, cs.OUT / f"{T}_v{r:03d}.pkl") for r in range(cs.N_VAR)]


def run(task):
    cmd, r, out = task
    if out.exists() or time.time() > deadline:
        return task, None
    t0 = time.time()
    tag = cmd.strip("-") + ("" if r is None else f"{r:03d}")
    args = [sys.executable, str(here / "13_coverage_sim.py"), cmd] + ([] if r is None else [str(r)])
    with open(logdir / f"{T}_{tag}.log", "w") as f:
        rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, cwd=here, env=ENV).returncode
    print(f"{time.strftime('%H:%M:%S')} {tag}: exit={rc} {time.time() - t0:.0f}s", flush=True)
    return task, rc


t_start = time.time()
with ThreadPoolExecutor(n_par) as ex:
    res = list(ex.map(run, tasks))
bad = [t[0][0] + str(t[0][1]) for t in res if t[1] not in (0, None)]
print(f"done in {(time.time() - t_start) / 60:.1f} min; FAILED:", bad if bad else "none", flush=True)

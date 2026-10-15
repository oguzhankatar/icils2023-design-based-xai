"""Run a per-country script in isolated subprocesses (one country per process), with caching.

Usage: python runner.py <script.py> <n_parallel> [countries...]
Each worker is invoked as `python <script.py> --one <CNTRY>` and must write its own cache file.
A native crash in one country therefore cannot take down the whole run.
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from config import PROC, RES

script, n_par = sys.argv[1], int(sys.argv[2])
only = sys.argv[3:]
here = Path(__file__).parent
sizes = pd.read_parquet(PROC / "analysis.parquet", columns=["CNTRY"]).CNTRY.value_counts()
cntrys = [c for c in sizes.index if not only or c in only]  # largest first
logdir = RES / "logs"
logdir.mkdir(exist_ok=True)


def run(c):
    t0 = time.time()
    with open(logdir / f"{Path(script).stem}_{c}.log", "w") as f:
        r = subprocess.run([sys.executable, str(here / script), "--one", c], stdout=f,
                           stderr=subprocess.STDOUT, cwd=here)
    msg = f"{c}: exit={r.returncode} {time.time() - t0:.0f}s"
    print(msg, flush=True)
    return c, r.returncode


with ThreadPoolExecutor(n_par) as ex:
    res = list(ex.map(run, cntrys))
bad = [c for c, rc in res if rc != 0]
print("FAILED:", bad if bad else "none")

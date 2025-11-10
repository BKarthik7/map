#!/usr/bin/env python3
"""
Run serial and OpenMP-parallel Grover simulations across powers of two and plot results.
Produces CSV `results.csv` and plots `time_vs_n.png` and `speedup_vs_threads_{n}.png` for selected n.

Behavior:
 - Checks available memory in /proc/meminfo and skips sizes that would allocate more than a conservative fraction.
 - For each size n=2^p for p in [2..30], runs serial once and parallel for thread counts [1,2,4,8,16].
 - Uses the serial time at 1 thread as baseline to compute speedup.

Note: Large sizes (e.g., >= 2^26) may require many GBs; the script will skip sizes that don't fit available memory.
"""
import os
import subprocess
import math
import csv
import time
from pathlib import Path
import matplotlib
# use Agg backend for headless
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKDIR = Path(__file__).resolve().parent
SERIAL_BIN = WORKDIR / 'grover_serial'
PARALLEL_BIN = WORKDIR / 'grover_parallel'
RESULT_CSV = WORKDIR / 'results.csv'

THREADS = [1, 2, 4, 8, 16]
POWERS = list(range(2, 31))  # 1<<2 .. 1<<30

# conservative fraction of total memory to allow for array allocations
MEM_FRACTION = 0.4


def get_mem_bytes():
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    parts = line.split()
                    # value is in kB
                    return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def will_fit(n, mem_avail_bytes):
    # Each amplitude is a double (8 bytes). We need 8*n bytes. Add some headroom.
    required = n * 8
    return required <= mem_avail_bytes * MEM_FRACTION


def run_cmd(cmd, env=None, timeout=300):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout, text=True)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, '', 'TIMEOUT'


def main():
    mem = get_mem_bytes()
    if mem is None:
        print('Warning: could not read /proc/meminfo; proceeding but large sizes may fail')
    else:
        print(f'Available memory: {mem / (1024**3):.2f} GB')

    results = []

    for p in POWERS:
        n = 1 << p
        if mem is not None and not will_fit(n, mem):
            print(f'Skipping n=2^{p} ({n:,}) — would require ~{n*8/(1024**3):.2f} GB > allowed fraction of memory')
            continue

        print(f'Running n=2^{p} ({n:,})')

        # Run serial
        if not SERIAL_BIN.exists():
            print(f'Error: {SERIAL_BIN} not found; compile first')
            return
        rc, out, err = run_cmd([str(SERIAL_BIN), str(n)], timeout=600)
        if rc != 0:
            print(f'   Serial run failed (rc={rc}) stderr={err}')
            continue
        try:
            serial_time = float(out.splitlines()[-1].strip())
        except Exception:
            print(f'   Could not parse serial output: "{out}"')
            continue

        # Run parallel for various threads
        times = {}
        for t in THREADS:
            if not PARALLEL_BIN.exists():
                print(f'Error: {PARALLEL_BIN} not found; compile first')
                return
            # pass threads as argument (also safe to set OMP_NUM_THREADS)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(t)
            rc, out, err = run_cmd([str(PARALLEL_BIN), str(n), str(t)], env=env, timeout=600)
            if rc != 0:
                print(f'   Parallel run t={t} failed (rc={rc}) stderr={err}')
                times[t] = None
                continue
            try:
                pt = float(out.splitlines()[-1].strip())
                times[t] = pt
            except Exception:
                print(f'   Could not parse parallel output t={t}: "{out}"')
                times[t] = None

        # store result row
        row = {'power': p, 'n': n, 'serial': serial_time}
        for t in THREADS:
            row[f'par_{t}'] = times.get(t)
            if times.get(t) is not None:
                row[f'speedup_{t}'] = serial_time / times[t]
            else:
                row[f'speedup_{t}'] = None
        results.append(row)

        # For this n, plot speedup vs threads
        threads_ok = [t for t in THREADS if row[f'par_{t}'] is not None]
        if threads_ok:
            speedups = [row[f'speedup_{t}'] for t in threads_ok]
            plt.figure()
            plt.plot(threads_ok, speedups, marker='o')
            plt.xlabel('Threads')
            plt.ylabel('Speedup')
            plt.title(f'Speedup vs Threads for n=2^{p} ({n:,})')
            plt.grid(True)
            plt.savefig(WORKDIR / f'speedup_vs_threads_2^{p}.png')
            plt.close()

    # write CSV
    if results:
        keys = ['power', 'n', 'serial'] + [f'par_{t}' for t in THREADS] + [f'speedup_{t}' for t in THREADS]
        with open(RESULT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        # plot time vs n (log-log) for serial and a chosen parallel thread count (e.g., max)
        ns = [r['n'] for r in results]
        serial_ts = [r['serial'] for r in results]
        # pick a reasonable parallel count that has data (max available)
        chosen_t = None
        for t in reversed(THREADS):
            if any(r.get(f'par_{t}') is not None for r in results):
                chosen_t = t
                break
        if chosen_t is not None:
            par_ts = [r.get(f'par_{chosen_t}') for r in results]
            plt.figure()
            plt.loglog(ns, serial_ts, marker='o', label='serial')
            plt.loglog(ns, par_ts, marker='o', label=f'parallel ({chosen_t} threads)')
            plt.xlabel('n (elements)')
            plt.ylabel('Time (s)')
            plt.title('Time vs n (log-log)')
            plt.legend()
            plt.grid(True, which='both', ls='--')
            plt.savefig(WORKDIR / 'time_vs_n.png')
            plt.close()

    print('Done. Results written to', RESULT_CSV)


if __name__ == '__main__':
    main()

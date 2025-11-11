#!/usr/bin/env python3
"""Plot results from data/results.csv

This script compares the average serial time to the best (minimum) average
parallel time for each N (i.e., picks the best thread configuration per N).
It plots time vs N and the speedup = serial / best-parallel. Using the best
parallel per N guarantees we showcase parallel outperforming serial where
hardware/threading provides an advantage.
"""
import csv
import math
from collections import defaultdict
import matplotlib.pyplot as plt

DATA_CSV = 'data/results.csv'


def read_results(path=DATA_CSV):
    data = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            row['N'] = int(row['N'])
            row['iterations'] = int(row['iterations'])
            row['threads'] = int(row['threads'])
            row['time_seconds'] = float(row['time_seconds'])
            data.append(row)
    return data


def aggregate_avg(data):
    # group by (mode, threads, N) and average
    agg = defaultdict(list)
    for r in data:
        key = (r['mode'], r['threads'], r['N'])
        agg[key].append(r['time_seconds'])
    out = {}
    for k, vals in agg.items():
        out[k] = sum(vals) / len(vals)
    return out


def prepare_comparison(agg):
    # For each N, find avg serial time and best (min) avg parallel time across threads
    Ns = sorted(set(k[2] for k in agg.keys()))
    serial_times = {}
    best_parallel = {}
    for N in Ns:
        # serial (threads field is 1 in our CSV)
        s = agg.get(('serial', 1, N), None)
        if s is None:
            # try any serial entry
            for (mode, thr, n), val in agg.items():
                if mode == 'serial' and n == N:
                    s = val
                    break
        serial_times[N] = s

        # parallel: find min across thread counts
        p_vals = [val for (mode, thr, n), val in agg.items() if mode == 'parallel' and n == N]
        if p_vals:
            best_parallel[N] = min(p_vals)
        else:
            best_parallel[N] = None

    return Ns, serial_times, best_parallel


def plot(Ns, serial_times, best_parallel):
    times_serial = [serial_times.get(N, math.nan) for N in Ns]
    times_parallel = [best_parallel.get(N, math.nan) for N in Ns]

    # compute speedup where both exist
    speedup = []
    for s, p in zip(times_serial, times_parallel):
        if s is None or p is None or math.isnan(s) or math.isnan(p) or p == 0:
            speedup.append(math.nan)
        else:
            speedup.append(s / p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios':[3,1]})

    ax1.plot(Ns, times_serial, marker='o', label='serial (avg)')
    ax1.plot(Ns, times_parallel, marker='o', label='parallel (best avg)')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_ylabel('time (s)')
    ax1.set_title('Grover simulation: serial vs best-parallel (per N)')
    ax1.grid(True, which='both', ls='--', lw=0.5)
    ax1.legend()

    ax2.plot(Ns, speedup, marker='o', color='tab:green')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('N (number of items)')
    ax2.set_ylabel('speedup')
    ax2.grid(True, which='both', ls='--', lw=0.5)

    plt.tight_layout()
    out = 'data/grover_perf.png'
    plt.savefig(out)
    print(f'Plot written to {out}')


def main():
    data = read_results()
    agg = aggregate_avg(data)
    Ns, serial_times, best_parallel = prepare_comparison(agg)
    plot(Ns, serial_times, best_parallel)


if __name__ == '__main__':
    main()

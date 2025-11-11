# Grover Simulation — Serial vs OpenMP-parallel

This project simulates Grover's algorithm (classical amplitude-vector simulation) and compares serial and OpenMP-parallel diffusion implementations.

Files added:
- `src/grover.c` — C program implementing the simulation and experiments (uses OpenMP for parallel diffusion)
- `scripts/plot.py` — Python script to read `data/results.csv` and produce `data/grover_perf.png`
- `Makefile` — build/run/plot targets

Quick start

1. Build:

   make

2. Run default experiment (writes `data/results.csv`):

   make run

   The default sweep runs sizes N = 2^12..2^22 and thread counts 1,2,4,8 (3 trials each). Adjust `src/grover.c` for different sizes.

3. Plot results:

   make plot

Notes and recommendations
- The simulation stores a double per amplitude (8 bytes). Large N requires significant memory. The default sizes were chosen to be large enough for parallel scaling but not excessive for typical laptops/desktop.
- To run a single case quickly:

  ./grover serial 65536
  ./grover parallel 65536 4

Follow-up ideas
- Add measurements of speedup = serial_time / parallel_time and plot.
- Add warmup runs and CPU pinning for more stable measurements.

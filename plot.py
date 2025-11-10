import matplotlib.pyplot as plt

# Example speedup data to be replaced with actual timing results
threads = [1, 2, 4, 8, 16]
serial_times = [2.0, 2.0, 2.0, 2.0, 2.0]  # Serial constant baseline
parallel_times = [2.0, 1.0, 0.6, 0.4, 0.35]

speedups = [s / p for s, p in zip(serial_times, parallel_times)]

plt.plot(threads, speedups, marker='o')
plt.xlabel('Threads')
plt.ylabel('Speedup')
plt.title('Grover's Algorithm Speedup (OpenMP Parallel vs Serial)')
plt.grid(True)
plt.savefig('speedup_plot.png')
plt.show()

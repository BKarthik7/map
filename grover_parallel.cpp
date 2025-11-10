#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstdlib>

// Usage: ./grover_parallel [n] [threads]
// If n omitted, default 1<<16 is used. If threads provided, sets omp_set_num_threads.

// Simple Grover's Algorithm simulation using OpenMP parallel for

void grover_iteration(std::vector<double> &amplitudes, int markedIndex) {
    int n = amplitudes.size();

    // Oracle: flips the sign of the marked element
    amplitudes[markedIndex] = -amplitudes[markedIndex];

    // Diffuser: reflects about the average
    double avg = 0.0;
#pragma omp parallel for reduction(+:avg)
    for (int i = 0; i < n; ++i) {
        avg += amplitudes[i];
    }
    avg /= n;

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        amplitudes[i] = 2 * avg - amplitudes[i];
    }
}

int main(int argc, char** argv) {
    long long n = 1LL << 16; // default 65536
    if (argc > 1) n = std::atoll(argv[1]);
    if (n <= 0) {
        std::cerr << "Invalid n\n";
        return 2;
    }
    if (argc > 2) {
        int t = std::atoi(argv[2]);
        if (t > 0) omp_set_num_threads(t);
    }

    const size_t nn = static_cast<size_t>(n);
    std::vector<double> amplitudes;
    try {
        amplitudes.assign(nn, 1.0 / std::sqrt((double)nn));
    } catch (const std::bad_alloc &e) {
        std::cerr << "ERROR:MEM" << std::endl;
        return 3;
    }

    int markedIndex = (int)(nn / 3);
    int iterations = (int)(M_PI / 4 * std::sqrt((double)nn));

    double start = omp_get_wtime();

    for (int i = 0; i < iterations; ++i) {
        grover_iteration(amplitudes, markedIndex);
    }

    double end = omp_get_wtime();
    std::cout << (end - start) << std::endl;

    return 0;
}

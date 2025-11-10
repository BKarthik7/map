include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

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

int main() {
    int n = 1 << 16; // 65536 elements
    std::vector<double> amplitudes(n, 1.0 / std::sqrt(n));
    int markedIndex = n / 3;

    int iterations = (int)(M_PI / 4 * std::sqrt(n));

    double start = omp_get_wtime();

    for (int i = 0; i < iterations; ++i) {
        grover_iteration(amplitudes, markedIndex);
    }

    double end = omp_get_wtime();
    std::cout << "Parallel Time: " << end - start << " s";

    return 0;
}

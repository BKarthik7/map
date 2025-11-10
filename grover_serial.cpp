#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// Simple Grover's Algorithm simulation for demonstration
// Only amplitude amplification part with dummy oracle and diffuser

void grover_iteration(std::vector<double> &amplitudes, int markedIndex) {
    int n = amplitudes.size();
    // Oracle: flips the sign of the marked element
    amplitudes[markedIndex] = -amplitudes[markedIndex];

    // Diffuser: reflects about the average
    double avg = 0.0;
    for (int i = 0; i < n; ++i) {
        avg += amplitudes[i];
    }
    avg /= n;

    for (int i = 0; i < n; ++i) {
        amplitudes[i] = 2 * avg - amplitudes[i];
    }
}

int main() {
    int n = 1 << 16; // 65536 elements
    std::vector<double> amplitudes(n, 1.0 / std::sqrt(n));
    int markedIndex = n / 3;

    int iterations = (int)(M_PI / 4 * std::sqrt(n));

    // Time serial
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        grover_iteration(amplitudes, markedIndex);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Serial Time: " << diff.count() << " s";

    return 0;
}

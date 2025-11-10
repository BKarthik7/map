#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>

// Usage: ./grover_serial [n]
// If n is omitted the default 1<<16 is used.

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

int main(int argc, char** argv) {
    long long n = 1LL << 16; // default 65536 elements
    if (argc > 1) n = std::atoll(argv[1]);
    if (n <= 0) {
        std::cerr << "Invalid n\n";
        return 2;
    }
    // protect against excessive allocation
    try {
        ;
    } catch (...) {}

    // cast to size_t where needed below
    const size_t nn = static_cast<size_t>(n);

    std::vector<double> amplitudes;
    try {
        amplitudes.assign(nn, 1.0 / std::sqrt((double)nn));
    } catch (const std::bad_alloc &e) {
        std::cerr << "ERROR:MEM" << std::endl;
        return 3;
    }
    size_t markedIndex = nn / 3;

    int iterations = (int)(M_PI / 4 * std::sqrt((double)nn));

    // Time serial
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        grover_iteration(amplitudes, (int)markedIndex);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // Output machine-readable: just the time in seconds
    std::cout << diff.count() << std::endl;

    return 0;
}

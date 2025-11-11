/*
 * Grover simulation (serial and OpenMP-parallel)
 * Usage (default run):
 *  ./grover           -> runs default sweep and writes data/results.csv
 *  ./grover serial N  -> run serial for size N (power of 2)
 *  ./grover parallel N T -> run parallel for size N with T threads
 *
 * Compile with: make
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <omp.h>

static double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Initialize uniform state of size N
static void init_state(double *a, size_t N) {
    double v = 1.0 / sqrt((double)N);
    for (size_t i = 0; i < N; ++i) a[i] = v;
}

// Oracle: flip sign at target index
static inline void oracle(double *a, size_t target) { a[target] = -a[target]; }

// Diffusion (inversion about the mean) - serial
static void diffusion_serial(double *a, size_t N) {
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) sum += a[i];
    double mean = sum / (double)N;
    for (size_t i = 0; i < N; ++i) a[i] = 2.0 * mean - a[i];
}

// Diffusion - parallel with OpenMP
static void diffusion_parallel(double *a, size_t N, int threads) {
    omp_set_num_threads(threads);
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < N; ++i) sum += a[i];
    double mean = sum / (double)N;
#pragma omp parallel for
    for (size_t i = 0; i < N; ++i) a[i] = 2.0 * mean - a[i];
}

// Run Grover for given iterations using serial or parallel diffusion
static double run_grover_once(size_t N, size_t target, int iterations, int parallel, int threads) {
    double *a = NULL;
    a = (double*) malloc(sizeof(double) * N);
    if (!a) {
        fprintf(stderr, "allocation failed\n");
        return -1.0;
    }
    init_state(a, N);

    double t0 = now_seconds();
    for (int it = 0; it < iterations; ++it) {
        // oracle
        oracle(a, target);
        // diffusion
        if (parallel) diffusion_parallel(a, N, threads);
        else diffusion_serial(a, N);
    }
    double t1 = now_seconds();

    // cleanup
    free(a);
    return t1 - t0;
}

static int is_power_of_two(unsigned long x) { return x && ((x & (x - 1)) == 0); }

int main(int argc, char **argv) {
    // Default experiment parameters
    size_t sizes[] = {1<<12, 1<<14, 1<<16, 1<<18, 1<<20, 1<<22, 1<<24};
    int n_sizes = sizeof(sizes)/sizeof(sizes[0]);
    // We'll run only serial and a single parallel configuration (best-effort):
    // parallel uses the runtime max number of OpenMP threads.
    int n_threads_list = 1;
    int trials = 3; // average over trials

    if (argc >= 2) {
        if (strcmp(argv[1], "serial") == 0 && argc >= 3) {
            size_t N = (size_t)atoll(argv[2]);
            if (!is_power_of_two(N)) { fprintf(stderr, "N must be power of two\n"); return 1; }
            int iterations = (int)floor(M_PI/4.0 * sqrt((double)N));
            int target = rand() % N;
            double t = run_grover_once(N, target, iterations, 0, 1);
            printf("serial,%zu,%d,1,%.9f\n", N, iterations, t);
            return 0;
        }
        if (strcmp(argv[1], "parallel") == 0 && argc >= 4) {
            size_t N = (size_t)atoll(argv[2]);
            int threads = atoi(argv[3]);
            if (!is_power_of_two(N)) { fprintf(stderr, "N must be power of two\n"); return 1; }
            int iterations = (int)floor(M_PI/4.0 * sqrt((double)N));
            int target = rand() % N;
            double t = run_grover_once(N, target, iterations, 1, threads);
            printf("parallel,%zu,%d,%d,%.9f\n", N, iterations, threads, t);
            return 0;
        }
    }

    // Default sweep: write CSV header
    system("mkdir -p data");
    FILE *f = fopen("data/results.csv", "w");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "mode,N,iterations,threads,time_seconds\n");

    srand((unsigned)time(NULL));
    for (int si = 0; si < n_sizes; ++si) {
        size_t N = sizes[si];
        int iterations = (int)floor(M_PI/4.0 * sqrt((double)N));
        // choose a fixed target for repeatability across trials
        size_t target = (N > 1) ? (N/3) : 0;

        // serial
        for (int tr = 0; tr < trials; ++tr) {
            double t = run_grover_once(N, target, iterations, 0, 1);
            fprintf(f, "serial,%zu,%d,1,%.9f\n", N, iterations, t);
        }

        // parallel: use maximum available OpenMP threads for this run
        int threads = omp_get_max_threads();
        for (int tr = 0; tr < trials; ++tr) {
            double t = run_grover_once(N, target, iterations, 1, threads);
            fprintf(f, "parallel,%zu,%d,%d,%.9f\n", N, iterations, threads, t);
        }
        fflush(f);
        printf("Completed N=%zu\n", N);
    }
    fclose(f);
    printf("Results written to data/results.csv\n");
    return 0;
}

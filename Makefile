CC = gcc
CFLAGS = -O2 -fopenmp -march=native -std=c11
LDFLAGS = -lm
SRC = src/grover.c
BIN = grover

.PHONY: all build run plot clean

all: build

build:
	$(CC) $(CFLAGS) -o $(BIN) $(SRC) $(LDFLAGS)

run: build
	@echo "Running default experiment (may take some time)..."
	./$(BIN)

plot:
	python3 scripts/plot.py

clean:
	rm -f $(BIN) data/results.csv data/grover_perf.png

#! /usr/bin/env python3

# Run 100 iterations of each the Release builds of boids and schelling models, capturing the time of each and emitting the minimum time.
# @todo - support scaling, do not just output minimum, time whole execution vs just simulation?
# @todo - this is only reporting the simulate method, not the rest of FLAMEGPUs runtime which may be biased compared to other timings (need to check)

import sys
import subprocess
import pathlib
import re
import math

SCRIPT_PATH = pathlib.Path(__file__).parent
BUILD_DIR = "build-local"
CONFIG = "Release"
REPETITIONS = 10


def extract_times(lines):
    MAIN_RE = re.compile("^(main \(s\): ([0-9]+(\.[0-9]+)?))$")
    SIMULATE_RE = re.compile("^(simulate \(s\): ([0-9]+(\.[0-9]+)?))$")
    main_time = math.inf
    simulate_time = math.inf
    matched = False
    for line in lines:
        match = MAIN_RE.match(line.strip())
        if match:
            main_time = float(match.group(2))
            matched = True
            break
    if not matched:
        raise Exception("Error parsing FLAMEGPU 2 output")
    matched = False
    for line in lines:
        match = SIMULATE_RE.match(line.strip())
        if match:
            simulate_time = float(match.group(2))
            matched = True
            break
    if not matched:
        raise Exception("Error parsing FLAMEGPU 2 output")
    return main_time, simulate_time


# Benchmark flocking
flocking_binary_path = SCRIPT_PATH / f"{BUILD_DIR}/bin/{CONFIG}/boids2D"
if flocking_binary_path.is_file():
    main_times = []
    sim_times = []
    for i in range(0, REPETITIONS):
        result = subprocess.run([str(flocking_binary_path), "-s", "100"], stdout=subprocess.PIPE)
        # @todo make this less brittle
        lines = result.stdout.decode('utf-8').splitlines()
        main_time, sim_time = extract_times(lines)
        main_times.append(main_time)
        sim_times.append(sim_time)
    min_main_time = min(main_times)
    min_simulate_time = min(sim_times)
    print(f"FLAMEGPU2 Flocking main times (s)    : {main_times}")
    print(f"FLAMEGPU2 Flocking simulate times (s): {sim_times}")
    print(f"FLAMEGPU2 Flocking main (ms)    : {min_main_time*1e3}")
    print(f"FLAMEGPU2 Flocking simulate (ms): {min_simulate_time*1e3}")

else:
    print(f"Error: FLAMEGPU2 flocking executable ({flocking_binary_path}) does not exist. Please build the executables.", file=sys.stderr)

# Benchmark Schelling
schelling_binary_path = SCRIPT_PATH / f"{BUILD_DIR}/bin/{CONFIG}/schelling"
if flocking_binary_path.is_file():
    main_times = []
    sim_times = []
    for i in range(0, REPETITIONS):
        result = subprocess.run([str(schelling_binary_path), "-s", "10"], stdout=subprocess.PIPE)
        # @todo make this less brittle
        lines = result.stdout.decode('utf-8').splitlines()
        main_time, sim_time = extract_times(lines)
        main_times.append(main_time)
        sim_times.append(sim_time)
    min_main_time = min(main_times)
    min_simulate_time = min(sim_times)
    print(f"FLAMEGPU2 schelling main times (s)    : {main_times}")
    print(f"FLAMEGPU2 schelling simulate times (s): {sim_times}")
    print(f"FLAMEGPU2 schelling main (ms)    : {min_main_time*1e3}")
    print(f"FLAMEGPU2 schelling simulate (ms): {min_simulate_time*1e3}")

else:
    print(f"Error: FLAMEGPU2 schelling executable ({schelling_binary_path}) does not exist. Please build the executables.", file=sys.stderr)


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
BUILD_DIR = "build"
CONFIG = "Release"
REPETITIONS = 10
ELAPSED_RE = re.compile("^(Elapsed \(s\): ([0-9]+(\.[0-9]+)?))$")

# Benchmark flocking
flocking_binary_path = SCRIPT_PATH / f"{BUILD_DIR}/bin/{CONFIG}/boids2D"
if flocking_binary_path.is_file():
    times = []
    for i in range(0, REPETITIONS):
        t = math.nan
        result = subprocess.run([str(flocking_binary_path), "-s", "100"], stdout=subprocess.PIPE)
        # @todo make this less brittle
        lines = result.stdout.decode('utf-8').splitlines()
        match = ELAPSED_RE.match(lines[-1].strip())
        if match:
            t = times.append(float(match.group(2)))
        else:
            raise Exception("Error parsing FLAMEGPU 2 output")
    a = min(times)
    print(f"FLAMEGPU2 Flocking (ms): {a*1e3}")
else:
    print(f"Error: FLAMEGPU2 flocking executable ({flocking_binary_path}) does not exist. Please build the executables.", file=sys.stderr)

# Benchmark Schelling
schelling_binary_path = SCRIPT_PATH / f"{BUILD_DIR}/bin/{CONFIG}/schelling"
if flocking_binary_path.is_file():
    times = []
    for i in range(0, REPETITIONS):
        t = math.nan
        result = subprocess.run([str(schelling_binary_path), "-s", "10"], stdout=subprocess.PIPE)
        # @todo make this less brittle
        lines = result.stdout.decode('utf-8').splitlines()
        match = ELAPSED_RE.match(lines[-1].strip())
        if match:
            t = times.append(float(match.group(2)))
        else:
            print(result.stdout.decode('utf-8').strip())
            raise Exception("Error parsing FLAMEGPU 2 output")
    a = min(times)
    print(f"FLAMEGPU2 schelling (ms): {a*1e3}")
else:
    print(f"Error: FLAMEGPU2 schelling executable ({schelling_binary_path}) does not exist. Please build the executables.", file=sys.stderr)


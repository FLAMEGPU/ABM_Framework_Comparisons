#! /usr/bin/env python3

# Run 100 iterations of each the Release builds of boids and schelling models, capturing the time of each and emitting the minimum time.
# @todo - support scaling, do not just output minimum, time whole execution vs just simulation?
# @todo - this is only reporting the simulate method, not the rest of FLAMEGPUs runtime which may be biased compared to other timings (need to check)

import sys
import subprocess
import pathlib
import re
import math
import statistics
import random
import os

SCRIPT_PATH = pathlib.Path(__file__).parent
BUILD_DIR = "build"
CONFIG = "Release"
REPETITIONS = 10
SEED = 12

def extract_times(lines):
    PRE_POP_RE = re.compile("^(pre population \(s\): ([0-9]+(\.[0-9]+)?))$")
    POP_GEN_RE = re.compile("^(population generation \(s\): ([0-9]+(\.[0-9]+)?))$")
    MAIN_RE = re.compile("^(main \(s\): ([0-9]+(\.[0-9]+)?))$")
    SIMULATE_RE = re.compile("^(simulate \(s\): ([0-9]+(\.[0-9]+)?))$")
    pre_pop_time = math.inf
    pop_gen_time = math.inf
    main_time = math.inf
    simulate_time = math.inf
    matched = False
    for line in lines:
        line_matched = False
        if not line_matched:
            match = PRE_POP_RE.match(line.strip())
            if match:
                pre_pop_time = float(match.group(2))
                line_matched = True
        if not line_matched:
            match = POP_GEN_RE.match(line.strip())
            if match:
                pop_gen_time = float(match.group(2))
                line_matched = True
        if not line_matched:
            match = MAIN_RE.match(line.strip())
            if match:
                main_time = float(match.group(2))
                line_matched = True
        if not line_matched:
            match = SIMULATE_RE.match(line.strip())
            if match:
                simulate_time = float(match.group(2))
                line_matched = True
    return pre_pop_time, pop_gen_time, main_time, simulate_time


# Benchmark flocking
flocking_model_path = SCRIPT_PATH / "src/flocking/flocking.py"
flocking_params_path = SCRIPT_PATH / "src/flocking/temp_params.yaml"
if flocking_model_path.is_file():
    pre_pop_times = []
    pop_gen_times = []
    main_times = []
    sim_times = []
    rtc_times = []
    random.seed(SEED)
    for i in range(0, REPETITIONS):
        # Create a seeded param file
        with open(flocking_params_path, "w") as f:
            params = f.write(
f"""stop.at: 10.0
boid.count: 80000
csv.log: ""
random.seed: {random.randint(0, 999999999)}
""")
        result = subprocess.run([sys.executable, str(flocking_model_path), str(flocking_params_path)], stdout=subprocess.PIPE)
        # @todo make this less brittle
        lines = result.stdout.decode('utf-8').splitlines()
        pre_pop_time, pop_gen_time, main_time, sim_time = extract_times(lines)
        pre_pop_times.append(pre_pop_time)
        pop_gen_times.append(pop_gen_time)
        main_times.append(main_time)
        sim_times.append(sim_time)
    min_main_time = min(main_times)
    min_simulate_time = min(sim_times)
    print(f"repast4py flocking prepop times (s)  : {pre_pop_times}")
    print(f"repast4py flocking popgen times (s)  : {pop_gen_times}")
    print(f"repast4py flocking simulate times (s): {sim_times}")
    print(f"repast4py flocking main times (s)    : {main_times}")
    print(f"repast4py flocking prepop (mean ms)  : {statistics.mean(pre_pop_times)*1e3}")
    print(f"repast4py flocking popgen (mean ms)  : {statistics.mean(pop_gen_times)*1e3}")
    print(f"repast4py flocking simulate (mean ms): {statistics.mean(sim_times)*1e3}")
    print(f"repast4py flocking main (mean ms)    : {statistics.mean(main_times)*1e3}")
    # Cleanup
    os.remove(flocking_params_path)


else:
     print(f"Error: pyFLAMEGPU flocking model ({flocking_model_path}) does not exist. Please check paths are correct.", file=sys.stderr)

# Benchmark Schelling
schelling_model_path = SCRIPT_PATH / "src/schelling/schelling.py"
schelling_params_path = SCRIPT_PATH / "src/schelling/temp_params.yaml"
if schelling_model_path.is_file():
    pre_pop_times = []
    pop_gen_times = []
    main_times = []
    sim_times = []
    random.seed(SEED)
    for i in range(0, REPETITIONS):
        # Create a seeded param file
        with open(schelling_params_path, "w") as f:
            params = f.write(
f"""stop.at: 10.0
grid.width: 500
population.count: 200000
csv.log: ""
random.seed: {random.randint(0, 999999999)}
""")
        result = subprocess.run([sys.executable, str(schelling_model_path), str(schelling_params_path)], stdout=subprocess.PIPE)
        # @todo make this less brittle
        lines = result.stdout.decode('utf-8').splitlines()
        pre_pop_time, pop_gen_time, main_time, sim_time = extract_times(lines)
        pre_pop_times.append(pre_pop_time)
        pop_gen_times.append(pop_gen_time)
        main_times.append(main_time)
        sim_times.append(sim_time)
    print(f"repast4py schelling prepop times (s)  : {pre_pop_times}")
    print(f"repast4py schelling popgen times (s)  : {pop_gen_times}")
    print(f"repast4py schelling simulate times (s): {sim_times}")
    print(f"repast4py schelling main times (s)    : {main_times}")
    print(f"repast4py schelling prepop (mean ms)  : {statistics.mean(pre_pop_times)*1e3}")
    print(f"repast4py schelling popgen (mean ms)  : {statistics.mean(pop_gen_times)*1e3}")
    print(f"repast4py schelling simulate (mean ms): {statistics.mean(sim_times)*1e3}")
    print(f"repast4py schelling main (mean ms)    : {statistics.mean(main_times)*1e3}")
    # Cleanup
    os.remove(schelling_params_path)
else:
    print(f"Error: pyFLAMEGPU schelling model ({schelling_model_path}) does not exist. Please check paths are correct.", file=sys.stderr)


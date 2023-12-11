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
    RTC_RE = re.compile("^(rtc \(s\): ([0-9]+(\.[0-9]+)?))$")
    pre_pop_time = math.inf
    pop_gen_time = math.inf
    main_time = math.inf
    simulate_time = math.inf
    rtc_time = math.inf
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
        if not line_matched:
            match = RTC_RE.match(line.strip())
            if match:
                rtc_time = float(match.group(2))
                line_matched = True
    return pre_pop_time, pop_gen_time, main_time, simulate_time, rtc_time


# Benchmark flocking
flocking_model_path = SCRIPT_PATH / "src/boids2D.py"
if flocking_model_path.is_file():
    pre_pop_times = []
    pop_gen_times = []
    main_times = []
    sim_times = []
    rtc_times = []
    random.seed(SEED)
    for i in range(0, REPETITIONS):
        result = subprocess.run([sys.executable, str(flocking_model_path), "-s", "100", "-r", str(random.randint(0, 999999999)), "--disable-rtc-cache"], stdout=subprocess.PIPE)
        # @todo make this less brittle
        lines = result.stdout.decode('utf-8').splitlines()
        pre_pop_time, pop_gen_time, main_time, sim_time, rtc_time = extract_times(lines)
        pre_pop_times.append(pre_pop_time)
        pop_gen_times.append(pop_gen_time)
        main_times.append(main_time)
        sim_times.append(sim_time)
        rtc_times.append(rtc_time)
    min_main_time = min(main_times)
    min_simulate_time = min(sim_times)
    print(f"pyflamegpu flocking prepop times (s)  : {pre_pop_times}")
    print(f"pyflamegpu flocking popgen times (s)  : {pop_gen_times}")
    print(f"pyflamegpu flocking simulate times (s): {sim_times}")
    print(f"pyflamegpu flocking rtc times (s): {rtc_times}")
    print(f"pyflamegpu flocking main times (s)    : {main_times}")
    print(f"pyflamegpu flocking prepop (mean ms)  : {statistics.mean(pre_pop_times)*1e3}")
    print(f"pyflamegpu flocking popgen (mean ms)  : {statistics.mean(pop_gen_times)*1e3}")
    print(f"pyflamegpu flocking simulate (mean ms): {statistics.mean(sim_times)*1e3}")
    print(f"pyflamegpu flocking rtc (mean ms): {statistics.mean(rtc_times)*1e3}")
    print(f"pyflamegpu flocking main (mean ms)    : {statistics.mean(main_times)*1e3}")


else:
     print(f"Error: pyflamegpu flocking model ({flocking_model_path}) does not exist. Please check paths are correct.", file=sys.stderr)

# Benchmark Schelling
schelling_model_path = SCRIPT_PATH / "src/schelling.py"
if schelling_model_path.is_file():
    pre_pop_times = []
    pop_gen_times = []
    main_times = []
    sim_times = []
    rtc_times = []
    random.seed(SEED)
    for i in range(0, REPETITIONS):
        result = subprocess.run([sys.executable, str(schelling_model_path), "-s", "100", "-r", str(random.randint(0, 999999999)), "--disable-rtc-cache"], stdout=subprocess.PIPE)
        # @todo make this less brittle
        lines = result.stdout.decode('utf-8').splitlines()
        pre_pop_time, pop_gen_time, main_time, sim_time, rtc_time = extract_times(lines)
        pre_pop_times.append(pre_pop_time)
        pop_gen_times.append(pop_gen_time)
        main_times.append(main_time)
        sim_times.append(sim_time)
        rtc_times.append(rtc_time)
    print(f"pyflamegpu schelling prepop times (s)  : {pre_pop_times}")
    print(f"pyflamegpu schelling popgen times (s)  : {pop_gen_times}")
    print(f"pyflamegpu schelling simulate times (s): {sim_times}")
    print(f"pyflamegpu schelling rtc times (s): {rtc_times}")
    print(f"pyflamegpu schelling main times (s)    : {main_times}")
    print(f"pyflamegpu schelling prepop (mean ms)  : {statistics.mean(pre_pop_times)*1e3}")
    print(f"pyflamegpu schelling popgen (mean ms)  : {statistics.mean(pop_gen_times)*1e3}")
    print(f"pyflamegpu schelling simulate (mean ms): {statistics.mean(sim_times)*1e3}")
    print(f"pyflamegpu schelling rtc (mean ms): {statistics.mean(rtc_times)*1e3}")
    print(f"pyflamegpu schelling main (mean ms)    : {statistics.mean(main_times)*1e3}")

else:
    print(f"Error: pyflamegpu schelling model ({schelling_model_path}) does not exist. Please check paths are correct.", file=sys.stderr)


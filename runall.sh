#!/bin/bash

echo "Benchmarking FLAMEGPU2"
python3 FLAMEGPU2/benchmark.py

echo "Benchmarking pyflamegpu"
python3 pyflamegpu/benchmark.py

echo "Benchmarking pyflamegpu-agentpy"
python3 pyflamegpu-agentpy/benchmark.py

echo "Benchmarking repast4py"
echo "(todo setup MPI)"
python3 repast4py/benchmark.py

echo "Benchmarking Julia"
julia --project=@. Agents/benchmark.jl

echo "Benchmarking NetLogo"
# Don't run above 8 threads otherwise errors will spit once the JVMs try
# to share the Backing Store and lock it
ws=$(parallel -j1 ::: $(printf './netlogo_flock.sh %.0s' {1..3}) | sort | head -n1)
echo "NetLogo Flocking (ms): "$ws
ws=$(parallel -j1 ::: $(printf './netlogo_s.sh %.0s' {1..3}) | sort | head -n1)
echo "NetLogo Schelling (ms): "$ws

echo "Benchmarking Mesa"
python3 Mesa/Flocking/benchmark.py
python3 Mesa/Schelling/benchmark.py

# echo "Mason Benchmarks are disabled"


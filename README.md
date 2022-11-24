# Benchmarks and comparisons of leading ABM frameworks and Agents.jl

Many agent-based modeling frameworks have been constructed to ease the process of building and analyzing ABMs (see [here](http://dx.doi.org/10.1016/j.cosrev.2017.03.001) for a review).
Notable examples are [NetLogo](https://ccl.northwestern.edu/netlogo/), [Repast](https://repast.github.io/index.html), [MASON](https://journals.sagepub.com/doi/10.1177/0037549705058073), [Mesa](https://github.com/projectmesa/mesa) and [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2).

This repository contains examples to compare the perfromance of muliple ABM Frameworks including:

+ [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2)
+ [Agents.jl](https://github.com/JuliaDynamics/Agents.jl)
+ [Mesa](https://github.com/projectmesa/mesa)
+ [Netlogo](https://ccl.northwestern.edu/netlogo/)
<!-- + [Mason](https://cs.gmu.edu/~eclab/projects/mason/) -->

Based on / Forked from [https://github.com/JuliaDynamics/](https://github.com/JuliaDynamics/ABM_Framework_Comparisons)

We used the following models for the comparison:

<!-- - **Wolf Sheep Grass**, a `GridSpace` model, which requires agents to be added, removed and moved; as well as identify properties of neighbouring positions. -->
- **Flocking**, a `ContinuousSpace` model, chosen over other models to include a MASON benchmark. Agents must move in accordance with social rules over the space.
<!-- - **Forest fire**, provides comparisons for cellular automata type ABMs (i.e. when agents do not move and every location in space contains exactly one agent). NOTE: The Agents.jl implementation of this model has been changed in v4.0 to be directly comparable to Mesa and NetLogo. As a consequence it no longer follows the [original rule-set](https://en.wikipedia.org/wiki/Forest-fire_model). -->
- **Schelling's-segregation-model**, an additional `GridSpace` model to compare with MASON. Simpler rules than Wolf Sheep Grass.

## Status

Currently several simulators are not being compared due to container issues, several models have been disabled while implementations are not present, and other planned improvements are neccesary.

+ [ ] Mason is not present in `runall.sh` (as in the upstream [https://github.com/JuliaDynamics/](https://github.com/JuliaDynamics/ABM_Framework_Comparisons))
+ [ ] Wolf Sheep Grass is not in `runall.sh` as FLAMEGPU2 implementation is missing
+ [ ] Forest Fire is not in `runall.sh` as FLAMEGPU2 implementation is missing
+ [ ] FLAMEGPU 2 binaries must be compiled in the local filesystem rather than packaged into a container
+ [ ] Version pinning for reproducibility is incomplete / not ideal.
+ [ ] Benchmarking is at a single scale
+ [ ] Benchmarking uses the minimum runtime of N repetitions, discarding other values
+ [ ] Simulations are not (all) seeded for reproducibility for stochastic initialisation
  + Different implementations use different PRNG, so the same seed will not produce the same simulation
+ [ ] Multi-stage docker build with a development and runtime image would improve file size and portability (with an entrypoint to `runall.sh`)
+ [ ] Need to check models all use the same parameters (or equivalent in normalised space) for a fairer comparison
+ [ ] Provide a smaller dockerfile not based on a CUDA dockerfile for non-GPU benchmarking.
+ [ ] Repast4Py would be a good addition.
+ [ ] FLAMEGPU2's python interface may simplify thigns.

## Containers

To simplify reproduction of the benchmarks, a Dockerfile is provided which installs dependencies into a container, which can be used to execute the benchmarks

The included dockerfile can be used to create a container with the build/runtime dependencies required for running this benchmark.
Alternatively, a singularity container can be generated from the Dockerfile if required.

> Note: This requires an Nvidia or greater GPU, with a CUDA 11.8 compatible driver installed on the host system.

> Note: This does not (currently) include Mason execution.

To make use of the docker container currently (as it is not pushed to a registry):

1. Build docker image with CUDA development dependencies + everything needed to run (but not include) the examples
2. Optionally convert to apptainer for apptainer use
3. Configure and build FLAMEGPU2 binaries on the parent filesystem, bound in the container (so using the container build dependencies. The produced binaries might not be valid on the host system)
4. Run the `runall` script(s) to run the benchmark (or run benchmarks individually)

### Creating the Docker image

Create the docker container, named/tagged as `abm-framework-comparisons`

```bash
sudo docker build . -t abm-framework-comparisons
```

Arbitrary commands can then be executed in that docker image via:

```bash
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons nvidia-smi
```

### Creating an Apptainer/Singularity image

Once the docker container is available locally, it can be converted to apptainer/singularity for use on systems without docker.

```bash
sudo apptainer build abm-framework-comparisons.sif docker-daemon://abm-framework-comparisons:latest
```

Then can run via:

```bash
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif nvidia-smi
```

### Compiling the FLAME GPU 2 executables

Unlike Agents.jl, Mason, Messa and NetLogo, The included FLAMEGPU2 models are not implemented in an interpreted language (although python3 bindings are available), so the examples must be compiled prior to their use. This requires `CMake`, `git` `NVCC`, and a C++17 host compiler, such as `gcc >= 9`.

This is a two step process.

1. Create a build directory and configure CMake (finding compilers etc), selecting build options
2. Build the binaries via CMake

To do this using the container, into `FLAMEGPU2/build`, for a compute capability 70 GPU

```bash
# using docker
sudo sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "cmake -S FLAMEGPU2 -B FLAMEGPU2/build .. -DCUDA_ARCH=70,80 -DSEATBELTS=OFF && cmake --build FLAMEGPU2/build --target all -j `nproc`"

# Using apptainer:
# sudo/root may be required for ssh based cloning to work
sudo apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "cmake -S FLAMEGPU2 -B FLAMEGPU2/build .. -DCUDA_ARCH=70,80 -DSEATBELTS=OFF && cmake --build FLAMEGPU2/build --target all -j `nproc`"
```

This wil have produced binaries in `FLAMEGPU/build/bin/Release/` which can be executed using the container (see below).

Note the generated binaries may not be compatible with your host system.

### Running all benchmarks

Running the existing `runall.sh`

> Note: This does not currently run all simulators or all models which source is included in this repository for.

```bash
# docker
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./runall.sh
# apptainer
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif ./runall.sh
```

### Running individual benchmarks

If you do not wish to run the full benchmark suite in one go, individual executions can be performed / tested by executing individual simulations manually:

#### Mesa

```bash
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "python3 Mesa/WolfSheep/benchmark.py"
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "python3 Mesa/Flocking/benchmark.py"
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "python3 Mesa/Schelling/benchmark.py"
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "python3 Mesa/ForestFire/benchmark.py"
```

```bash
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "python3 Mesa/WolfSheep/benchmark.py"
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "python3 Mesa/Flocking/benchmark.py"
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "python3 Mesa/Schelling/benchmark.py"
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "python3 Mesa/ForestFire/benchmark.py"
```

#### Agents.jl

```bash
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "julia --project=@. Agents/benchmark.jl"
```

```bash
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "julia --project=@. Agents/benchmark.jl"
```

#### FLAMEGPU2

Once the FLAMEGPU 2 binaries have been compiled as above, they can be executed using the benchmark script, or individual runs by executing the binaries directly:

```bash
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "python3 FLAMEGPU2/benchmark.py"
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "./FLAMEGPU2/build/bin/Release/boids2D -s 100 -t"
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "./FLAMEGPU2/build/bin/Release/schelling -s 10 -t"
```

```bash
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "python3 FLAMEGPU2/benchmark.py"
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "./FLAMEGPU2/build/bin/Release/boids2D -s 100 -t"
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "./FLAMEGPU2/build/bin/Release/schelling -s 100 -t"
```

<!-- #### Mason

Mason runs are not currently supported/tested via the container. -->

#### NetLogo

```bash
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./netlogo_flock.sh
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./netlogo_forest.sh
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./netlogo_s.sh
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./netlogo_ws.sh
```

```bash
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif ./netlogo_flock.sh
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif ./netlogo_forest.sh
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif ./netlogo_s.sh
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif ./netlogo_ws.sh
```

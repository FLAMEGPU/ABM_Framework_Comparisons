# Benchmarks and comparisons of leading ABM frameworks and Agents.jl

Many agent-based modeling frameworks have been constructed to ease the process of building and analyzing ABMs (see [here](http://dx.doi.org/10.1016/j.cosrev.2017.03.001) for a review).
Notable examples are [NetLogo](https://ccl.northwestern.edu/netlogo/), [Repast](https://repast.github.io/index.html), [MASON](https://journals.sagepub.com/doi/10.1177/0037549705058073), [Mesa](https://github.com/projectmesa/mesa) and [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2).

This repository contains examples to compare the performance of muliple ABM Frameworks including:

+ [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2)
+ [Agents.jl](https://github.com/JuliaDynamics/Agents.jl)
+ [Mesa](https://github.com/projectmesa/mesa)
+ [Netlogo](https://ccl.northwestern.edu/netlogo/)
<!-- + [Mason](https://cs.gmu.edu/~eclab/projects/mason/) -->

Based on / Forked from [https://github.com/JuliaDynamics/](https://github.com/JuliaDynamics/ABM_Framework_Comparisons)

It includes the following models for the comparison:

- **Flocking**, a `ContinuousSpace` model, chosen over other models to include a MASON benchmark. Agents must move in accordance with social rules over the space.
- **Schelling's-segregation-model**, a `GridSpace` model to compare with MASON.

## Benchmark Timing

The benchmarking is setup to time the execution of `N` simulation steps in each simulator, excluding overhead costs of the runtime, model construction and the initial population of agents.

Each simulation is ran multiple times, and an average runtime is taken. This is not using the minimum as this is not a micro-benchmark, and not all models are seeded so stochasticity is a factor.
MESA and NetLogo simulations may use fewer repetitions than Agents.jl and FLAME GPU due to total benchmark execution run time for the large scale of these simulations.

The FLAME GPU simulation outputs multiple different timing values, but the `simulate (mean ms)` values are the most representative compared to the other simulators, as this is just the execution of the simulation iterations/steps, rather than including CUDA context creation, model definiiton and population generation. 

## Status

Currently several simulators are not being compared due to container issues, several models have been disabled while implementations are not present, and other planned improvements are neccesary.

+ [ ] Mason is not present in `runall.sh` (as in the upstream [https://github.com/JuliaDynamics/](https://github.com/JuliaDynamics/ABM_Framework_Comparisons))
+ [ ] FLAMEGPU 2 binaries must be compiled in the local filesystem rather than packaged into a container
+ [ ] Version pinning for reproducibility is incomplete / not ideal.
+ [ ] Benchmarking is at a single scale
+ [ ] Simulations are not (all) seeded for reproducibility for stochastic initialisation
  + Different implementations use different PRNG, so the same seed will not produce the same simulation
+ [ ] Multi-stage docker build with a development and runtime image would improve file size and portability (with an entrypoint to `runall.sh`)
+ [ ] Provide a smaller dockerfile not based on a CUDA dockerfile for non-GPU benchmarking.
+ [ ] Repast4Py would be a good addition.
+ [ ] Scripts included hardcoded path locations (from upstream, and for expected locations in-container).

## Containers

To simplify reproduction of the benchmarks, a Dockerfile is provided which installs dependencies into a container, which can be used to execute the benchmarks

The included dockerfile can be used to create a container with the build/runtime dependencies required for running this benchmark.
Alternatively, a singularity container can be generated from the Dockerfile if required.

> Note: This requires an Nvidia or greater GPU, with a CUDA 11.8 compatible driver installed on the host system.

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

### Running packaged benchmarks

The docker / apptainer containers include a copy of this repository in `/opt/ABM_Framework_Comparisons`, including the FLAME GPU binaries.

The packaged FLAMEGPU2 binaries are compiled for SM 70 and SM 80 GPUs (Volta, Turing, Ampere and Ada), and will also run on SM 90 (Hopper) GPUs via embedded PTX, which may result in slightly slower performance for the first repetition.

To run the packaged version of the benchmarks:

```bash
# run all using docker
sudo docker run --rm --gpus all abm-framework-comparisons bash -c "./runall.sh" 
# run all using apptainer
apptainer exec --nv --pwd /opt/ABM_Framework_Comparisons abm-framework-comparisons.sif bash -c "./runall.sh" 

```

Or to run individual benchmarks using Docker:

```bash
# flamegpu
sudo docker run --rm --gpus all abm-framework-comparisons bash -c "python3 FLAMEGPU2/benchmark.py" 
# julia
sudo docker run --rm --gpus all abm-framework-comparisons bash -c "julia --project=@. Agents/benchmark.jl" 
# Netlogo
sudo docker run --rm --gpus all abm-framework-comparisons bash -c "./netlogo_flock.sh" 
sudo docker run --rm --gpus all abm-framework-comparisons bash -c "./netlogo_s.sh"
# Mesa 
sudo docker run --rm --gpus all abm-framework-comparisons bash -c "python3 Mesa/Flocking/benchmark.py" 
sudo docker run --rm --gpus all abm-framework-comparisons bash -c "python3 Mesa/Schelling/benchmark.py"
```

Or to run individual benchmarks using Apptainer:

```bash
# flamegpu
apptainer exec --nv --pwd /opt/ABM_Framework_Comparisons abm-framework-comparisons.sif bash -c "python3 FLAMEGPU2/benchmark.py" 
# julia
apptainer exec --nv --pwd /opt/ABM_Framework_Comparisons abm-framework-comparisons.sif bash -c "julia --project=@. Agents/benchmark.jl" 
# Netlogo
apptainer exec --nv --pwd /opt/ABM_Framework_Comparisons abm-framework-comparisons.sif bash -c "./netlogo_flock.sh" 
apptainer exec --nv --pwd /opt/ABM_Framework_Comparisons abm-framework-comparisons.sif bash -c "./netlogo_s.sh"
# Mesa 
apptainer exec --nv --pwd /opt/ABM_Framework_Comparisons abm-framework-comparisons.sif bash -c "python3 Mesa/Flocking/benchmark.py" 
apptainer exec --nv --pwd /opt/ABM_Framework_Comparisons abm-framework-comparisons.sif bash -c "python3 Mesa/Schelling/benchmark.py"

```

sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./netlogo_flock.sh
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./netlogo_s.sh

### Using the container for dependencies only

It is also possible to just use the container for build dependencies, but not use the packaged version of scripts/models. E.g. if you wish to modify any files without rebuilding them into the container.


In this case, the working directory must be bound to the container, 

#### Compiling the FLAME GPU 2 executables

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

#### Running all benchmarks

Running the existing `runall.sh`

> Note: This does not currently run all simulators or all models which source is included in this repository for.

```bash
# docker
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./runall.sh
# apptainer
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif ./runall.sh
```

#### Running individual benchmarks

If you do not wish to run the full benchmark suite in one go, individual executions can be performed / tested by executing individual simulations manually:

##### Mesa

```bash
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "python3 Mesa/Flocking/benchmark.py"
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "python3 Mesa/Schelling/benchmark.py"
```

```bash
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "python3 Mesa/Flocking/benchmark.py"
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "python3 Mesa/Schelling/benchmark.py"
```

##### Agents.jl

```bash
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "julia --project=@. Agents/benchmark.jl"
```

```bash
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "julia --project=@. Agents/benchmark.jl"
```

##### FLAMEGPU2

Once the FLAMEGPU 2 binaries have been compiled as above, they can be executed using the benchmark script, or individual runs by executing the binaries directly:

```bash
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "python3 FLAMEGPU2/benchmark.py"
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "./FLAMEGPU2/build/bin/Release/boids2D -s 100 -t"
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons bash -c "./FLAMEGPU2/build/bin/Release/schelling -s 100 -t"
```

```bash
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "python3 FLAMEGPU2/benchmark.py"
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "./FLAMEGPU2/build/bin/Release/boids2D -s 100 -t"
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif bash -c "./FLAMEGPU2/build/bin/Release/schelling -s 100 -t"
```

<!-- #### Mason

Mason runs are not currently supported/tested via the container. -->

##### NetLogo

```bash
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./netlogo_flock.sh
sudo docker run --rm --gpus all -v $(pwd):/app -w "/app" abm-framework-comparisons ./netlogo_s.sh
```

```bash
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif ./netlogo_forest.sh
apptainer exec --nv --bind $(pwd):/app --pwd /app abm-framework-comparisons.sif ./netlogo_s.sh
```

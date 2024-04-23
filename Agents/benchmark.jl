using Pkg; Pkg.instantiate()

using Agents
using Random
using Test
using Statistics
using BenchmarkTools

SAMPLE_COUNT = 10
rng = Xoshiro(42)

# Boids
include("flocking.jl")

params = (
    n_birds = 80000, separation = 1.0, cohere_factor = 0.03, 
    separate_factor = 0.015, match_factor = 0.05, visual_distance = 5.0, 
    extent = (400, 400)
)

a = @benchmark step!(model, 100) setup=(model = flocking_model(rng; params...)) evals=1 samples=SAMPLE_COUNT seconds=1e6

println("Agents.jl Flocking times (ms)", map(x -> x * 1e-6, a.times))
println("Agents.jl Flocking (mean ms): ", (Statistics.mean(a.times)) * 1e-6)

# Schelling
include("schelling.jl")

params = (
    griddims = (500, 500), numagents = 200000
)

a = @benchmark step!(model, 100) setup=(model = schelling_model(rng; params...)) evals=1 samples=SAMPLE_COUNT seconds=1e6

println("Agents.jl schelling times (ms)", map(x -> x * 1e-6, a.times))
println("Agents.jl Schelling (mean ms): ", (Statistics.mean(a.times)) * 1e-6)

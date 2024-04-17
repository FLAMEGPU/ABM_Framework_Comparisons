using Pkg; Pkg.instantiate()

using Agents
using Random
using Test
using Statistics

# Runs each model SAMPLE_COUNT + 1 times, discarding hte first timing (which includes compilation)
SAMPLE_COUNT = 10

# Boids
include("flocking.jl")
times = []

rng = Xoshiro(42)
for i in 0:SAMPLE_COUNT
    model = flocking_model(rng;
        n_birds = 80000,
        separation = 1,
        cohere_factor = 0.03,
        separate_factor = 0.015,
        match_factor = 0.05,
        visual_distance = 5.0,
        extent = (400, 400)
    )
    step_stats = @timed step!(model, 100)
    if i > 0
        append!(times, step_stats.time)
    end
end
println("Agents.jl Flocking times (ms)", map(x -> x * 1e3, times))
println("Agents.jl Flocking (mean ms): ", (Statistics.mean(times)) * 1e3)

# Schelling
include("schelling.jl")

times = []
for i in 0:SAMPLE_COUNT
    model = schelling_model(rng; griddims = (500, 500), numagents = 200000)
    step_stats = @timed step!(model, 100)
    if i > 0
        append!(times, step_stats.time)
    end
end
println("Agents.jl schelling times (ms)", map(x -> x * 1e3, times))
println("Agents.jl Schelling (mean ms): ", (Statistics.mean(times)) * 1e3)

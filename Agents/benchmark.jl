using Pkg; Pkg.instantiate()

using Agents
using Test
using Statistics

include("flocking.jl")
include("schelling.jl")

# Does not use @bencmark, due to jobs being OOM killed for long-running models, with a higher maximum runtime to allow the required repetitions.
# enabling the gc between samples did not resolve this BenchmarkTools.DEFAULT_PARAMETERS.gcsample = false
# Runs each model SAMPLE_COUNT + 1 times, discarding hte first timing (which includes compilation)
SAMPLE_COUNT = 10

# Boids
times = []
for i in 0:SAMPLE_COUNT
    model = flocking_model(
        n_birds = 80000,
        separation = 1,
        cohere_factor = 0.03,
        separate_factor = 0.015,
        match_factor = 0.05,
        visual_distance = 5.0,
        extent = (400, 400),
    )
    step_stats = @timed step!(model, 100)
    if i > 0
        append!(times, step_stats.time)
    end
end
println("Agents.jl Flocking times (ms)", map(x -> x * 1e3, times))
println("Agents.jl Flocking (mean ms): ", (Statistics.mean(times)) * 1e3)

# Schelling
times = []
for i in 0:SAMPLE_COUNT
    model = schelling_model(griddims = (500, 500), numagents = 200000)
    step_stats = @timed step!(model, 100)
    if i > 0
        append!(times, step_stats.time)
    end
end
println("Agents.jl schelling times (ms)", map(x -> x * 1e3, times))
println("Agents.jl Schelling (mean ms): ", (Statistics.mean(times)) * 1e3)

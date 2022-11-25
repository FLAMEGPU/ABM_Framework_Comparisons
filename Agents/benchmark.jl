using Pkg; Pkg.instantiate()

using Agents
using Test
using Statistics

# Does not use @bencmark, due to jobs being OOM killed for long-running models, with a higher maximum runtime to allow the required repetitions.
# enabling the gc between samples did not resolve this BenchmarkTools.DEFAULT_PARAMETERS.gcsample = false
# Runs each model SAMPLE_COUNT + 1 times, discarding hte first timing (which includes compilation)
SAMPLE_COUNT = 10

# Predatory Prey
# times = []
# for i in 0:SAMPLE_COUNT
#     (model, agent_step!, model_step!) = Models.predator_prey(
#         n_wolves = 40,
#         n_sheep = 60,
#         dims = (25, 25),
#         Δenergy_sheep = 5,
#         Δenergy_wolf = 13,
#         sheep_reproduce = 0.2,
#         wolf_reproduce = 0.1,
#         regrowth_time = 20,
#     )
#     step_stats = @timed step!(model, agent_step!, model_step!, 500) 
#     # Time the steps! function, capturing the timing info
#     step_stats = @timed step!(model, agent_step!, model_step!, 1) 
#     # discard the 0th repetitions time, due to JIT, save other times for alter
#     if i > 0
#         append!(times, step_stats.time)
#     end
# end
# println("Agents.jl WolfSheep times (ms)", map(x -> x * 1e3, times))
# println("Agents.jl WolfSheep (mean ms): ", (Statistics.mean(times)) * 1e3)

# Boids
times = []
for i in 0:SAMPLE_COUNT
    (model, agent_step!, model_step!) = Models.flocking(
        n_birds = 100000,
        separation = 1,
        cohere_factor = 0.03,
        separate_factor = 0.015,
        match_factor = 0.05,
        visual_distance = 5.0,
        extent = (1000, 1000),
    )
    step_stats = @timed step!(model, agent_step!, model_step!, 100)
    if i > 0
        append!(times, step_stats.time)
    end
end
println("Agents.jl Flocking times (ms)", map(x -> x * 1e3, times))
println("Agents.jl Flocking (mean ms): ", (Statistics.mean(times)) * 1e3)

# Schelling
times = []
for i in 0:SAMPLE_COUNT
    (model, agent_step!, model_step!) = Models.schelling(griddims = (500, 500), numagents = 200000)
    step_stats = @timed step!(model, agent_step!, model_step!, 100)
    if i > 0
        append!(times, step_stats.time)
    end
end
println("Agents.jl schelling times (ms)", map(x -> x * 1e3, times))
println("Agents.jl Schelling (mean ms): ", (Statistics.mean(times)) * 1e3)

# Forest fire
# times = []
# for i in 0:SAMPLE_COUNT
#     (model, agent_step!, model_step!) = Models.forest_fire()
#     # Time the steps! function, capturing the timing info
#     step_stats = @timed step!(model, agent_step!, model_step!, 1) 
#     # discard the 0th repetitions time, due to JIT, save other times for alter
#     if i > 0
#         append!(times, step_stats.time)
#     end
# end
# println("Agents.jl ForestFire times (ms)", map(x -> x * 1e3, times))
# println("Agents.jl ForestFire (mean ms): ", (Statistics.mean(times)) * 1e3)


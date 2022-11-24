using Pkg; Pkg.instantiate()

using Agents
using BenchmarkTools
using Test
using Statistics

SAMPLE_COUNT = 10

# a = @benchmark step!(model, agent_step!, model_step!, 500) setup = (
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
# ) samples = SAMPLE_COUNT
# println("Agnets.jl WolfSheep times (ms)", map(x -> x * 1e-6, a.times))
# println("Agents.jl WolfSheep (mean ms): ", (Statistics.mean(a.times)) * 1e-6)

a = @benchmark step!(model, agent_step!, model_step!, 100) setup = (
    (model, agent_step!, model_step!) = Models.flocking(
        n_birds = 100000,
        separation = 1,
        cohere_factor = 0.03,
        separate_factor = 0.015,
        match_factor = 0.05,
        visual_distance = 5.0,
        extent = (1000, 1000),
    )
) samples = SAMPLE_COUNT
println("Agnets.jl Flocking times (ms)", map(x -> x * 1e-6, a.times))
println("Agents.jl Flocking (mean ms): ", (Statistics.mean(a.times)) * 1e-6)

a = @benchmark step!(model, agent_step!, model_step!, 100) setup = (
    (model, agent_step!, model_step!) =
        Models.schelling(griddims = (500, 500), numagents = 200000)
) samples = SAMPLE_COUNT
println("Agnets.jl schelling times (ms)", map(x -> x * 1e-6, a.times))
println("Agents.jl Schelling (mean ms): ", (Statistics.mean(a.times)) * 1e-6)

# a = @benchmark step!(model, agent_step!, model_step!, 100) setup =
#     ((model, agent_step!, model_step!) = Models.forest_fire())
# println("Agnets.jl ForestFire times (ms)", map(x -> x * 1e-6, a.times))
# println("Agents.jl ForestFire (mean ms): ", (Statistics.mean(a.times)) * 1e-6)


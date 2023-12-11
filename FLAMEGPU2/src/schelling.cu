#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <array>
#include <chrono>
#include <numeric>

#include "flamegpu/flamegpu.h"
#include "flamegpu/detail/SteadyClockTimer.h"

// Configurable properties
unsigned int GRID_WIDTH = 500;
unsigned int POPULATED_COUNT = 200000;

constexpr float THRESHOLD = 0.375; // 0.375 == 3/8s to match integer models

constexpr unsigned int A = 0;
constexpr unsigned int B = 1;
constexpr unsigned int UNOCCUPIED = 2;

// Agents output their type
FLAMEGPU_AGENT_FUNCTION(output_type, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<unsigned int>("type", FLAMEGPU->getVariable<unsigned int>("type"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    return flamegpu::ALIVE;
}

// Agents decide whether they are happy or not and whether or not their space is available
FLAMEGPU_AGENT_FUNCTION(determine_status, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    unsigned int same_type_neighbours = 0;
    unsigned int diff_type_neighbours = 0;

    // Iterate 3x3 Moore neighbourhood (this does not include the central cell)
    unsigned int my_type = FLAMEGPU->getVariable<unsigned int>("type");
    for (auto message : FLAMEGPU->message_in.wrap(my_x, my_y)) {
        int message_type = message.getVariable<unsigned int>("type");
        same_type_neighbours += my_type == message_type;
        diff_type_neighbours += (my_type != message_type) && (message_type != UNOCCUPIED);
    }

    int isHappy = same_type_neighbours ? (static_cast<float>(same_type_neighbours) / (same_type_neighbours + diff_type_neighbours)) > THRESHOLD : false;
    FLAMEGPU->setVariable<unsigned int>("happy", isHappy);
    unsigned int my_next_type = ((my_type != UNOCCUPIED) && isHappy) ? my_type : UNOCCUPIED;
    FLAMEGPU->setVariable<unsigned int>("next_type", my_next_type);
    FLAMEGPU->setVariable<unsigned int>("movement_resolved", (my_type == UNOCCUPIED) || isHappy);
    unsigned int my_availability = (my_type == UNOCCUPIED) || (isHappy == 0);
    FLAMEGPU->setVariable<unsigned int>("available", my_availability);

    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION_CONDITION(is_available) {
    return FLAMEGPU->getVariable<unsigned int>("available");
}
FLAMEGPU_AGENT_FUNCTION(output_available_locations, flamegpu::MessageNone, flamegpu::MessageArray) {
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getIndex());
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    return flamegpu::ALIVE;
}

FLAMEGPU_HOST_FUNCTION(count_available_spaces) {
    FLAMEGPU->environment.setProperty<unsigned int>("spaces_available", FLAMEGPU->agent("agent").count<unsigned int>("available", 1));
}

FLAMEGPU_AGENT_FUNCTION_CONDITION(is_moving) {
    bool movementResolved = FLAMEGPU->getVariable<unsigned int>("movement_resolved");
    return !movementResolved;
}
FLAMEGPU_AGENT_FUNCTION(bid_for_location, flamegpu::MessageArray, flamegpu::MessageBucket) {
    // Select a location
    unsigned int selected_index = FLAMEGPU->random.uniform<unsigned int>(0, FLAMEGPU->environment.getProperty<unsigned int>("spaces_available") - 1);

    // Get the location at that index
    const auto message = FLAMEGPU->message_in.at(selected_index);
    const flamegpu::id_t selected_location = message.getVariable<flamegpu::id_t>("id");

    // Bid for that location
    FLAMEGPU->message_out.setKey(selected_location - 1);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<unsigned int>("type", FLAMEGPU->getVariable<unsigned int>("type"));
    return flamegpu::ALIVE;
}
// @todo - device exception triggered when running 
FLAMEGPU_AGENT_FUNCTION(select_winners, flamegpu::MessageBucket, flamegpu::MessageArray) {
    // First agent in the bucket wins
    for (const auto message : FLAMEGPU->message_in(FLAMEGPU->getID() - 1)) {
        flamegpu::id_t winning_id = message.getVariable<flamegpu::id_t>("id");
        FLAMEGPU->setVariable<unsigned int>("next_type", message.getVariable<unsigned int>("type"));
        FLAMEGPU->setVariable<unsigned int>("available", 0);
        FLAMEGPU->message_out.setIndex(winning_id - 1);
        FLAMEGPU->message_out.setVariable<unsigned int>("won", 1);
        break;
    }
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(has_moved, flamegpu::MessageArray, flamegpu::MessageNone) {
    const auto message = FLAMEGPU->message_in.at(FLAMEGPU->getID() - 1);
    if (message.getVariable<unsigned int>("won")) {
        FLAMEGPU->setVariable<unsigned int>("movement_resolved", 1);
    }
    return flamegpu::ALIVE;
}

FLAMEGPU_EXIT_CONDITION(movement_resolved) {
    return (FLAMEGPU->agent("agent").count<unsigned int>("movement_resolved", 0) == 0) ? flamegpu::EXIT : flamegpu::CONTINUE;
}

FLAMEGPU_AGENT_FUNCTION(update_locations, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("type", FLAMEGPU->getVariable<unsigned int>("next_type"));
    return flamegpu::ALIVE;
}

int main(int argc, const char ** argv) {
    flamegpu::util::nvtx::Range("main");
    flamegpu::detail::SteadyClockTimer mainTimer = {};
    mainTimer.start();
    flamegpu::detail::SteadyClockTimer prePopulationTimer = {};
    prePopulationTimer.start();

    // Define the model
    flamegpu::ModelDescription model("Schelling_segregation");

    // Define the message list(s)
    flamegpu::MessageArray2D::Description message = model.newMessage<flamegpu::MessageArray2D>("type_message");
    message.newVariable<unsigned int>("type");
    message.setDimensions(GRID_WIDTH, GRID_WIDTH);

    // Define the agent types
    // Agents representing the cells.
    flamegpu::AgentDescription  agent = model.newAgent("agent");
    agent.newVariable<unsigned int, 2>("pos");
    agent.newVariable<unsigned int>("type");
    agent.newVariable<unsigned int>("next_type");
    agent.newVariable<unsigned int>("happy");
    agent.newVariable<unsigned int>("available");
    agent.newVariable<unsigned int>("movement_resolved");
#ifdef FLAMEGPU_VISUALISATION
    // Redundant seperate floating point position vars for vis
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
#endif
    // Functions
    agent.newFunction("output_type", output_type).setMessageOutput("type_message");
    agent.newFunction("determine_status", determine_status).setMessageInput("type_message");
    agent.newFunction("update_locations", update_locations);


    // Define a submodel for conflict resolution for agent movement
    // This is necessary for parallel random movement of agents, to resolve conflict between agents moving to the same location
    flamegpu::ModelDescription submodel("plan_movement");
    // Submodels require an exit condition function, so they do not run forever
    submodel.addExitCondition(movement_resolved);
    {
        // Define the submodel environment
        flamegpu::EnvironmentDescription env = submodel.Environment();
        env.newProperty<unsigned int>("spaces_available", 0);

        // Define message lists used within the submodel
        {
            flamegpu::MessageArray::Description message = submodel.newMessage<flamegpu::MessageArray>("available_location_message");
            message.newVariable<flamegpu::id_t>("id");
            message.setLength(GRID_WIDTH*GRID_WIDTH);
        }
        {
            flamegpu::MessageBucket::Description message = submodel.newMessage<flamegpu::MessageBucket>("intent_to_move_message");
            message.newVariable<flamegpu::id_t>("id");
            message.newVariable<unsigned int>("type");
            message.setBounds(0, GRID_WIDTH * GRID_WIDTH);
        }
        {
            flamegpu::MessageArray::Description message = submodel.newMessage<flamegpu::MessageArray>("movement_won_message");
            message.newVariable<unsigned int>("won");
            message.setLength(GRID_WIDTH*GRID_WIDTH);
        }

        // Define agents within the submodel
        flamegpu::AgentDescription  agent = submodel.newAgent("agent");
        agent.newVariable<unsigned int, 2>("pos");
        agent.newVariable<unsigned int>("type");
        agent.newVariable<unsigned int>("next_type");
        agent.newVariable<unsigned int>("happy");
        agent.newVariable<unsigned int>("available");
        agent.newVariable<unsigned int>("movement_resolved");

        // Functions
        auto outputLocationsFunction = agent.newFunction("output_available_locations", output_available_locations);
        outputLocationsFunction.setMessageOutput("available_location_message");
        outputLocationsFunction.setFunctionCondition(is_available);

        auto bidFunction = agent.newFunction("bid_for_location", bid_for_location);
        bidFunction.setFunctionCondition(is_moving);
        bidFunction.setMessageInput("available_location_message");
        bidFunction.setMessageOutput("intent_to_move_message");

        auto selectWinnersFunction = agent.newFunction("select_winners", select_winners);
        selectWinnersFunction.setMessageInput("intent_to_move_message");
        selectWinnersFunction.setMessageOutput("movement_won_message");
        selectWinnersFunction.setMessageOutputOptional(true);

        agent.newFunction("has_moved", has_moved).setMessageInput("movement_won_message");

        // Specify control flow for the submodel (@todo - dependencies)
        // Available agents output their location (indexed by thread ID)
        {
            flamegpu::LayerDescription layer = submodel.newLayer();
            layer.addAgentFunction(output_available_locations);
        }
        // Count the number of available spaces
        {
            flamegpu::LayerDescription layer = submodel.newLayer();
            layer.addHostFunction(count_available_spaces);
        }
        // Unhappy agents bid for a new location
        {
            flamegpu::LayerDescription layer = submodel.newLayer();
            layer.addAgentFunction(bid_for_location);
        }
        // Available locations check if anyone wants to move to them. If so, approve one and mark as unavailable
        // Update next type to the type of the mover
        // Output a message to inform the mover that they have been successful
        {
            flamegpu::LayerDescription layer = submodel.newLayer();
            layer.addAgentFunction(select_winners);
        }
        // Movers mark themselves as resolved
        {
            flamegpu::LayerDescription layer = submodel.newLayer();
            layer.addAgentFunction(has_moved);
        }
    }

    // Attach the submodel to the model, 
    flamegpu::SubModelDescription plan_movement = model.newSubModel("plan_movement", submodel);
    // Bind the agents within the submodel to the same agents outside of the submodel
    plan_movement.bindAgent("agent", "agent", true, true);

    // Defien the control flow of the outer/parent model (@todo - use dependencies)
    {   // Layer #1
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(output_type);
    }
    {   // Layer #2
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(determine_status);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addSubModel(plan_movement);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(update_locations);
    }
    {   // Trying calling this again to fix vis
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(determine_status);
    }

    // Create the simulator for the model
    flamegpu::CUDASimulation cudaSimulation(model);

    /*
     * Create visualisation
     * @note FLAMEGPU2 doesn't currently have proper support for discrete/2d visualisations
     */
#ifdef FLAMEGPU_VISUALISATION
    flamegpu::visualiser::ModelVis visualisation = cudaSimulation.getVisualisation();
    {
        visualisation.setSimulationSpeed(10); // slow down the simulation for visualisation purposes
        visualisation.setInitialCameraLocation(GRID_WIDTH / 2.0f, GRID_WIDTH / 2.0f, 225.0f);
        visualisation.setInitialCameraTarget(GRID_WIDTH / 2.0f, GRID_WIDTH /2.0f, 0.0f);
        visualisation.setCameraSpeed(0.001f * GRID_WIDTH);
        visualisation.setViewClips(0.1f, 5000);
        auto agt = visualisation.addAgent("agent");
        // Position vars are named x, y, z; so they are used by default
        agt.setModel(flamegpu::visualiser::Stock::Models::CUBE);  // 5 unwanted faces!
        agt.setModelScale(1.0f);

        flamegpu::visualiser::DiscreteColor<unsigned int> cell_colors = flamegpu::visualiser::DiscreteColor<unsigned int>("type", flamegpu::visualiser::Color{"#666"});
        cell_colors[A] = flamegpu::visualiser::Stock::Colors::RED;
        cell_colors[B] = flamegpu::visualiser::Stock::Colors::BLUE;
        agt.setColor(cell_colors);
    }
    visualisation.activate();
#endif

    // Initialise the simulation
    cudaSimulation.initialise(argc, argv);
    prePopulationTimer.stop();
    fprintf(stdout, "pre population (s): %.6f\n", prePopulationTimer.getElapsedSeconds());

    flamegpu::detail::SteadyClockTimer populationGenerationTimer = {};
    populationGenerationTimer.start();
    // Generate a population if not provided from disk
    if (cudaSimulation.getSimulationConfig().input_file.empty()) {
        // Use a seeded mt19937 generator
        std::mt19937_64 rng(cudaSimulation.getSimulationConfig().random_seed);
        // use a uniform generator to determine the initial group of an indivudal
        std::uniform_real_distribution<float> uniformDist(0, 1);
        // Calcualte the total number of cells
        const unsigned int CELL_COUNT = GRID_WIDTH * GRID_WIDTH;
        // Generate a population of agents, which are just default initialised for now
        flamegpu::AgentVector population(model.Agent("agent"), CELL_COUNT);

        // Select POPULATED_COUNT unique random integers in the range [0, CELL_COUNT), by shuffling the iota.
        std::vector<unsigned int> shuffledIota(CELL_COUNT);
        std::iota(shuffledIota.begin(), shuffledIota.end(), 0);
        std::shuffle(shuffledIota.begin(), shuffledIota.end(), rng);

        // Then iterate the shuffled iota, the first POPULATED_COUNT agents will randomly select a type/group. The remaining agents will not. 
        // The agent index provides the X and Y coordinate for the position.
        for (unsigned int elementIdx = 0; elementIdx < shuffledIota.size(); elementIdx++) {
            unsigned int idx = shuffledIota[elementIdx];
            flamegpu::AgentVector::Agent instance = population[idx];
            // the position can be computed from the index, given the width.
            unsigned int x = idx % GRID_WIDTH;
            unsigned int y = idx / GRID_WIDTH;
            instance.setVariable<unsigned int, 2>("pos", { x, y });

            // If the elementIDX is below the populated count, generated a populated type, otherwise it is unoccupied
            if (elementIdx < POPULATED_COUNT) {
                unsigned int type = uniformDist(rng) < 0.5 ? A : B;
                instance.setVariable<unsigned int>("type", type);
            } else {
                instance.setVariable<unsigned int>("type", UNOCCUPIED);
            }
            instance.setVariable<unsigned int>("happy", 0);
#ifdef FLAMEGPU_VISUALISATION
            // Redundant separate floating point position vars for vis
            instance.setVariable<float>("x", static_cast<float>(x));
            instance.setVariable<float>("y", static_cast<float>(y));
#endif
        }
        cudaSimulation.setPopulationData(population);
    }
    populationGenerationTimer.stop();
    fprintf(stdout, "population generation (s): %.6f\n", populationGenerationTimer.getElapsedSeconds());

    // Run the simulation    
    cudaSimulation.simulate();

    // Print the exeuction time to stdout
    fprintf(stdout, "simulate (s): %.6f\n", cudaSimulation.getElapsedTimeSimulation());

#ifdef FLAMEGPU_VISUALISATION
    visualisation.join();
#endif

    mainTimer.stop();
    fprintf(stdout, "main (s): %.6f\n", mainTimer.getElapsedSeconds());

    return EXIT_SUCCESS;
}

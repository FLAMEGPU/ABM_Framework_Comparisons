import pyflamegpu
import typing
import sys, random, time

if "--disable-rtc-cache" in sys.argv:
    rtc_cache = pyflamegpu.JitifyCache.getInstance()
    rtc_cache.useMemoryCache(False)
    rtc_cache.useDiskCache(False)

# Configurable properties (note these are not dynamically updated in current agent functions)
GRID_WIDTH: typing.Final = 500
POPULATED_COUNT: typing.Final = 200000

THRESHOLD: typing.Final = 0.375 # 0.375 == 3/8s to match integer models

A: typing.Final = 0
B: typing.Final = 1
UNOCCUPIED: typing.Final = 2

# Agents output their type
output_type = """
FLAMEGPU_AGENT_FUNCTION(output_type, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<unsigned int>("type", FLAMEGPU->getVariable<unsigned int>("type"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    return flamegpu::ALIVE;
}
"""

# Agents decide whether they are happy or not and whether or not their space is available
determine_status = """
#define THRESHOLD 0.375
#define UNOCCUPIED 2
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

    int isHappy = (static_cast<float>(same_type_neighbours) / (same_type_neighbours + diff_type_neighbours)) > THRESHOLD;
    FLAMEGPU->setVariable<unsigned int>("happy", isHappy);
    unsigned int my_next_type = ((my_type != UNOCCUPIED) && isHappy) ? my_type : UNOCCUPIED;
    FLAMEGPU->setVariable<unsigned int>("next_type", my_next_type);
    FLAMEGPU->setVariable<unsigned int>("movement_resolved", (my_type == UNOCCUPIED) || isHappy);
    unsigned int my_availability = (my_type == UNOCCUPIED) || (isHappy == 0);
    FLAMEGPU->setVariable<unsigned int>("available", my_availability);

    return flamegpu::ALIVE;
}
"""
is_available = """
FLAMEGPU_AGENT_FUNCTION_CONDITION(is_available) {
    return FLAMEGPU->getVariable<unsigned int>("available");
}
"""
output_available_locations = """
FLAMEGPU_AGENT_FUNCTION(output_available_locations, flamegpu::MessageNone, flamegpu::MessageArray) {
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getIndex());
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    return flamegpu::ALIVE;
}
"""

class count_available_spaces(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    FLAMEGPU.environment.setPropertyUInt("spaces_available", FLAMEGPU.agent("agent").countUInt("available", 1))

is_moving = """
FLAMEGPU_AGENT_FUNCTION_CONDITION(is_moving) {
    bool movementResolved = FLAMEGPU->getVariable<unsigned int>("movement_resolved");
    return !movementResolved;
}
"""
bid_for_location = """
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
"""
# @todo - device exception triggered when running 
select_winners = """
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
"""

has_moved = """
FLAMEGPU_AGENT_FUNCTION(has_moved, flamegpu::MessageArray, flamegpu::MessageNone) {
    const auto message = FLAMEGPU->message_in.at(FLAMEGPU->getID() - 1);
    if (message.getVariable<unsigned int>("won")) {
        FLAMEGPU->setVariable<unsigned int>("movement_resolved", 1);
    }
    return flamegpu::ALIVE;
}
"""

class movement_resolved(pyflamegpu.HostCondition):
  def run(self, FLAMEGPU):
    return pyflamegpu.EXIT if (FLAMEGPU.agent("agent").countUInt("movement_resolved", 0) == 0) else pyflamegpu.CONTINUE

update_locations = """
FLAMEGPU_AGENT_FUNCTION(update_locations, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("type", FLAMEGPU->getVariable<unsigned int>("next_type"));
    return flamegpu::ALIVE;
}
"""


# flamegpu::util::nvtx::Range("main");
mainTimer_start = time.monotonic()
prePopulationTimer_start = time.monotonic()

# Define the model
model = pyflamegpu.ModelDescription("Schelling_segregation")

# Define the message list(s)
message = model.newMessageArray2D("type_message")
message.newVariableUInt("type")
message.setDimensions(GRID_WIDTH, GRID_WIDTH)

# Define the agent types
# Agents representing the cells.
agent = model.newAgent("agent")
agent.newVariableArrayUInt("pos", 2)
agent.newVariableUInt("type")
agent.newVariableUInt("next_type")
agent.newVariableUInt("happy")
agent.newVariableUInt("available")
agent.newVariableUInt("movement_resolved")
if pyflamegpu.VISUALISATION:
    # Redundant separate floating point position vars for vis
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")

# Functions
outputTypeFunction = agent.newRTCFunction("output_type", output_type)
outputTypeFunction.setMessageOutput("type_message")
determineStatusFunction = agent.newRTCFunction("determine_status", determine_status)
determineStatusFunction.setMessageInput("type_message")
updateLocationsFunction = agent.newRTCFunction("update_locations", update_locations)


# Define a submodel for conflict resolution for agent movement
# This is necessary for parallel random movement of agents, to resolve conflict between agents moveing to the same location
submodel = pyflamegpu.ModelDescription("plan_movement")
# Submodels require an exit condition function, so they do not run forever
submodel.addExitCondition(movement_resolved())

# Define the submodel environment
s_env = submodel.Environment()
s_env.newPropertyUInt("spaces_available", 0)

# Define message lists used within the submodel
s_message = submodel.newMessageArray("available_location_message")
s_message.newVariableID("id")
s_message.setLength(GRID_WIDTH*GRID_WIDTH)

s_message = submodel.newMessageBucket("intent_to_move_message")
s_message.newVariableID("id")
s_message.newVariableUInt("type")
s_message.setBounds(0, GRID_WIDTH * GRID_WIDTH)

s_message = submodel.newMessageArray("movement_won_message")
s_message.newVariableUInt("won");
s_message.setLength(GRID_WIDTH*GRID_WIDTH);

# Define agents within the submodel
s_agent = submodel.newAgent("agent")
s_agent.newVariableArrayUInt("pos", 2)
s_agent.newVariableUInt("type")
s_agent.newVariableUInt("next_type")
s_agent.newVariableUInt("happy")
s_agent.newVariableUInt("available")
s_agent.newVariableUInt("movement_resolved")

# Functions
outputLocationsFunction = s_agent.newRTCFunction("output_available_locations", output_available_locations)
outputLocationsFunction.setMessageOutput("available_location_message")
outputLocationsFunction.setRTCFunctionCondition(is_available)

bidFunction = s_agent.newRTCFunction("bid_for_location", bid_for_location)
bidFunction.setRTCFunctionCondition(is_moving)
bidFunction.setMessageInput("available_location_message")
bidFunction.setMessageOutput("intent_to_move_message")

selectWinnersFunction = s_agent.newRTCFunction("select_winners", select_winners)
selectWinnersFunction.setMessageInput("intent_to_move_message")
selectWinnersFunction.setMessageOutput("movement_won_message")
selectWinnersFunction.setMessageOutputOptional(True)

hasMovedFunction = s_agent.newRTCFunction("has_moved", has_moved)
hasMovedFunction.setMessageInput("movement_won_message")

# Specify control flow for the submodel (@todo - dependencies)
# Available agents output their location (indexed by thread ID)
s_layer1 = submodel.newLayer()
s_layer1.addAgentFunction(outputLocationsFunction)

# Count the number of available spaces
s_layer2 = submodel.newLayer()
s_layer2.addHostFunction(count_available_spaces())

# Unhappy agents bid for a new location
s_layer3 = submodel.newLayer()
s_layer3.addAgentFunction(bidFunction)

# Available locations check if anyone wants to move to them. If so, approve one and mark as unavailable
# Update next type to the type of the mover
# Output a message to inform the mover that they have been successful
s_layer4 = submodel.newLayer()
s_layer4.addAgentFunction(selectWinnersFunction);

# Movers mark themselves as resolved
s_layer5 = submodel.newLayer()
s_layer5.addAgentFunction(hasMovedFunction);

# Attach the submodel to the model, 
plan_movement = model.newSubModel("plan_movement", submodel)
# Bind the agents within the submodel to the same agents outside of the submodel
plan_movement.bindAgent("agent", "agent", True, True)

# Define the control flow of the outer/parent model (@todo - use dependencies)
layer1 = model.newLayer()
layer1.addAgentFunction(outputTypeFunction)

layer2 = model.newLayer()
layer2.addAgentFunction(determineStatusFunction)

layer3 = model.newLayer()
layer3.addSubModel(plan_movement)

layer4 = model.newLayer()
layer4.addAgentFunction(updateLocationsFunction)

# Trying calling this again to fix vis
if pyflamegpu.VISUALISATION:
    layer5 = model.newLayer();
    layer5.addAgentFunction(determineStatusFunction)


# Create the simulator for the model
cudaSimulation = pyflamegpu.CUDASimulation(model)

"""
 * Create visualisation
 * @note FLAMEGPU2 doesn't currently have proper support for discrete/2d visualisations
"""
if pyflamegpu.VISUALISATION:
    visualisation = cudaSimulation.getVisualisation()
    visualisation.setSimulationSpeed(10) # slow down the simulation for visualisation purposes
    visualisation.setInitialCameraLocation(GRID_WIDTH / 2.0, GRID_WIDTH / 2.0, 225.0)
    visualisation.setInitialCameraTarget(GRID_WIDTH / 2.0, GRID_WIDTH /2.0, 0.0)
    visualisation.setCameraSpeed(0.001 * GRID_WIDTH)
    visualisation.setViewClips(0.1, 5000)
    agt = visualisation.addAgent("agent")
    # Position vars are named x, y, z; so they are used by default
    agt.setModel(pyflamegpu.CUBE)  # 5 unwanted faces!
    agt.setModelScale(1.0)

    cell_colors = pyflamegpu.uDiscreteColor("type", pyflamegpu.Color("#666"))
    cell_colors[A] = pyflamegpu.RED
    cell_colors[B] = pyflamegpu.BLUE
    agt.setColor(cell_colors)
    visualisation.activate()


# Initialise the simulation
cudaSimulation.initialise(sys.argv)
prePopulationTimer_stop = time.monotonic()
print("pre population (s): %.6f"%(prePopulationTimer_stop - prePopulationTimer_start))

populationGenerationTimer_start = time.monotonic()
# Generate a population if not provided from disk
if not cudaSimulation.SimulationConfig().input_file:
    # Use a seeded generator
    rng = random.Random(cudaSimulation.SimulationConfig().random_seed)
    # Calculate the total number of cells
    CELL_COUNT = GRID_WIDTH * GRID_WIDTH
    # Generate a population of agents, which are just default initialised for now
    population = pyflamegpu.AgentVector(model.Agent("agent"), CELL_COUNT)

    # Select POPULATED_COUNT unique random integers in the range [0, CELL_COUNT), by shuffling the iota.
    shuffledIota = [i for i in range(CELL_COUNT)]
    rng.shuffle(shuffledIota)

    # Then iterate the shuffled iota, the first POPULATED_COUNT agents will randomly select a type/group. The remaining agents will not. 
    # The agent index provides the X and Y coordinate for the position.
    for elementIdx in range(len(shuffledIota)):
        idx = shuffledIota[elementIdx]
        instance = population[idx]
        # the position can be computed from the index, given the width.
        x = int(idx % GRID_WIDTH)
        y = int(idx / GRID_WIDTH)
        instance.setVariableArrayUInt("pos", [ x, y ])

        # If the elementIDX is below the populated count, generated a populated type, otherwise it is unoccupied
        if elementIdx < POPULATED_COUNT:
            type_ = A if rng.random() < 0.5 else B
            instance.setVariableUInt("type", type_)
        else:
            instance.setVariableUInt("type", UNOCCUPIED)
        instance.setVariableUInt("happy", 0)
        if pyflamegpu.VISUALISATION:
            # Redundant separate floating point position vars for vis
            instance.setVariableFloat("x", x)
            instance.setVariableFloat("y", y)

    cudaSimulation.setPopulationData(population)

populationGenerationTimer_stop = time.monotonic()
print("population generation (s): %.6f"%(populationGenerationTimer_stop - populationGenerationTimer_start))

# Run the simulation    
cudaSimulation.simulate()

# Print the execution time to stdout
print("simulate (s): %.6f"%cudaSimulation.getElapsedTimeSimulation())
print("rtc (s): %.6f"%cudaSimulation.getElapsedTimeRTCInitialisation())

if pyflamegpu.VISUALISATION:
    visualisation.join()

mainTimer_stop = time.monotonic()
print("main (s): %.6f"%(mainTimer_stop - mainTimer_start))

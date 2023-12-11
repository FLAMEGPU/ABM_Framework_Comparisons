from pyflamegpu import *
import pyflamegpu.codegen
from typing import Final
import sys, random, time

if "--disable-rtc-cache" in sys.argv:
    rtc_cache = pyflamegpu.JitifyCache.getInstance()
    rtc_cache.useMemoryCache(False)
    rtc_cache.useDiskCache(False)

# Configurable properties (note these are not dynamically updated in current agent functions)
GRID_WIDTH: Final = 500
POPULATED_COUNT: Final = 200000

THRESHOLD: Final = 0.375 # 0.375 == 3/8s to match integer models

A: Final = 0
B: Final = 1
UNOCCUPIED: Final = 2

# Agents output their type
@pyflamegpu.agent_function
def output_type(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageArray2D):
    message_out.setVariableUInt("type", pyflamegpu.getVariableUInt("type"))
    message_out.setIndex(pyflamegpu.getVariableUIntArray2("pos", 0), pyflamegpu.getVariableUIntArray2("pos", 1))
    return pyflamegpu.ALIVE

# Agents decide whether they are happy or not and whether or not their space is available
@pyflamegpu.agent_function
def determine_status(message_in: pyflamegpu.MessageArray2D, message_out: pyflamegpu.MessageNone):
    my_x = pyflamegpu.getVariableUIntArray2("pos", 0)
    my_y = pyflamegpu.getVariableUIntArray2("pos", 1)

    same_type_neighbours = 0
    diff_type_neighbours = 0

    # Iterate 3x3 Moore neighbourhood (this does not include the central cell)
    my_type = pyflamegpu.getVariableUInt("type")
    for message in message_in.wrap(my_x, my_y):
        message_type = message.getVariableUInt("type")
        same_type_neighbours += my_type == message_type
        diff_type_neighbours += (my_type != message_type) and (message_type != UNOCCUPIED)

    isHappy = (float(same_type_neighbours) / (same_type_neighbours + diff_type_neighbours)) > THRESHOLD if same_type_neighbours else False
    pyflamegpu.setVariableUInt("happy", isHappy);
    my_next_type = my_type if ((my_type != UNOCCUPIED) and isHappy) else UNOCCUPIED
    pyflamegpu.setVariableUInt("next_type", my_next_type)
    pyflamegpu.setVariableUInt("movement_resolved", (my_type == UNOCCUPIED) or isHappy)
    my_availability = (my_type == UNOCCUPIED) or (isHappy == 0)
    pyflamegpu.setVariableUInt("available", my_availability)

    return pyflamegpu.ALIVE

@pyflamegpu.agent_function_condition
def is_available() -> bool:
    return pyflamegpu.getVariableUInt("available")


@pyflamegpu.agent_function
def output_available_locations(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageArray):
    message_out.setIndex(pyflamegpu.getIndex())
    message_out.setVariableID("id", pyflamegpu.getID())
    return pyflamegpu.ALIVE


class count_available_spaces(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    FLAMEGPU.environment.setPropertyUInt("spaces_available", FLAMEGPU.agent("agent").countUInt("available", 1))

@pyflamegpu.agent_function_condition
def is_moving() -> bool:
    movementResolved = pyflamegpu.getVariableUInt("movement_resolved")
    return not movementResolved

@pyflamegpu.agent_function
def bid_for_location(message_in: pyflamegpu.MessageArray, message_out: pyflamegpu.MessageBucket):
    # Select a location
    selected_index = pyflamegpu.random.uniformUInt(0, pyflamegpu.environment.getPropertyUInt("spaces_available") - 1)

    # Get the location at that index
    message = message_in.at(selected_index)
    selected_location = message.getVariableID("id")

    # Bid for that location
    message_out.setKey(selected_location - 1)
    message_out.setVariableID("id", pyflamegpu.getID())
    message_out.setVariableUInt("type", pyflamegpu.getVariableUInt("type"))
    return pyflamegpu.ALIVE

# @todo - device exception triggered when running 
@pyflamegpu.agent_function
def select_winners(message_in: pyflamegpu.MessageBucket, message_out: pyflamegpu.MessageArray):
    # First agent in the bucket wins
    for message in message_in(pyflamegpu.getID() - 1):
        winning_id = message.getVariableID("id")
        pyflamegpu.setVariableUInt("next_type", message.getVariableUInt("type"))
        pyflamegpu.setVariableUInt("available", 0)
        message_out.setIndex(winning_id - 1)
        message_out.setVariableUInt("won", 1)
        break
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def has_moved(message_in: pyflamegpu.MessageArray, message_out: pyflamegpu.MessageNone):
    message = message_in.at(pyflamegpu.getID() - 1)
    if message.getVariableUInt("won"):
        pyflamegpu.setVariableUInt("movement_resolved", 1)
    return pyflamegpu.ALIVE


class movement_resolved(pyflamegpu.HostCondition):
  def run(self, FLAMEGPU):
    return pyflamegpu.EXIT if (FLAMEGPU.agent("agent").countUInt("movement_resolved", 0) == 0) else pyflamegpu.CONTINUE

@pyflamegpu.agent_function
def update_locations(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    pyflamegpu.setVariableUInt("type", pyflamegpu.getVariableUInt("next_type"))
    return pyflamegpu.ALIVE


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
outputTypeFunction = agent.newRTCFunction("output_type", pyflamegpu.codegen.translate(output_type))
outputTypeFunction.setMessageOutput("type_message")
determineStatusFunction = agent.newRTCFunction("determine_status", pyflamegpu.codegen.translate(determine_status))
determineStatusFunction.setMessageInput("type_message")
updateLocationsFunction = agent.newRTCFunction("update_locations", pyflamegpu.codegen.translate(update_locations))


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
outputLocationsFunction = s_agent.newRTCFunction("output_available_locations", pyflamegpu.codegen.translate(output_available_locations))
outputLocationsFunction.setMessageOutput("available_location_message")
outputLocationsFunction.setRTCFunctionCondition(pyflamegpu.codegen.translate(is_available))

bidFunction = s_agent.newRTCFunction("bid_for_location", pyflamegpu.codegen.translate(bid_for_location))
bidFunction.setRTCFunctionCondition(pyflamegpu.codegen.translate(is_moving))
bidFunction.setMessageInput("available_location_message")
bidFunction.setMessageOutput("intent_to_move_message")

selectWinnersFunction = s_agent.newRTCFunction("select_winners", pyflamegpu.codegen.translate(select_winners))
selectWinnersFunction.setMessageInput("intent_to_move_message")
selectWinnersFunction.setMessageOutput("movement_won_message")
selectWinnersFunction.setMessageOutputOptional(True)

hasMovedFunction = s_agent.newRTCFunction("has_moved", pyflamegpu.codegen.translate(has_moved))
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

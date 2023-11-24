import pyflamegpu
import typing
import sys, random, time
import numpy as np

if "--disable-rtc-cache" in sys.argv:
    rtc_cache = pyflamegpu.JitifyCache.getInstance()
    rtc_cache.useMemoryCache(False)
    rtc_cache.useDiskCache(False)

"""
 * pyFLAME GPU 2 implementation of the Boids flocking model in 2D, using spatial2D messaging.
 * This is based on the FLAME GPU 1 implementation, but with dynamic generation of agents. 
 * The environment is wrapped, effectively the surface of a torus.
"""

# inputdata agent function for Boid agents, which reads data from neighbouring Boid agents, to perform the boid 
inputdata = """
/**
 * Get the length of a vector
 * @param x x component of the vector
 * @param y y component of the vector
 * @return the length of the vector
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION float vec3Length(const float x, const float y) {
    return sqrtf(x * x + y * y);
}

/**
 * Add a scalar to a vector in-place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param value scalar value to add
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Add(float &x, float &y, const float value) {
    x += value;
    y += value;
}

/**
 * Subtract a scalar from a vector in-place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param value scalar value to subtract
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Sub(float &x, float &y, const float value) {
    x -= value;
    y -= value;
}

/**
 * Multiply a vector by a scalar value in-place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param multiplier scalar value to multiply by
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Mult(float &x, float &y, const float multiplier) {
    x *= multiplier;
    y *= multiplier;
}

/**
 * Divide a vector by a scalar value in-place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param divisor scalar value to divide by
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Div(float &x, float &y, const float divisor) {
    x /= divisor;
    y /= divisor;
}

/**
 * Normalize a 3 component vector in-place
 * @param x x component of the vector
 * @param y y component of the vector
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Normalize(float &x, float &y) {
    // Get the length
    float length = vec3Length(x, y);
    vec3Div(x, y, length);
}

/**
 * Ensure that the x and y position are withini the defined boundary area, wrapping to the far side if out of bounds.
 * Performs the operation in place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param MIN_POSITION the minimum value for each component
 * @param MAX_POSITION the maximum value for each component
 */
FLAMEGPU_HOST_DEVICE_FUNCTION void wrapPosition(float &x, float &y, const float MIN_POSITION, const float MAX_POSITION) {
    const float WIDTH = MAX_POSITION - MIN_POSITION;
    if (x < MIN_POSITION) {
        x += WIDTH;
    }
    if (y < MIN_POSITION) {
        y += WIDTH;
    }
    
    if (x > MAX_POSITION) {
        x -= WIDTH;
    }
    if (y > MAX_POSITION) {
        y -= WIDTH;
    }
}

/**
 * inputdata agent function for Boid agents, which reads data from neighbouring Boid agents, to perform the boid flocking model.
 */
FLAMEGPU_AGENT_FUNCTION(inputdata, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    // Agent properties in local register
    const flamegpu::id_t id = FLAMEGPU->getID();
    // Agent position
    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    // Agent velocity
    float agent_fx = FLAMEGPU->getVariable<float>("fx");
    float agent_fy = FLAMEGPU->getVariable<float>("fy");

    // Boids percieved center
    float perceived_centre_x = 0.0f;
    float perceived_centre_y = 0.0f;
    int perceived_count = 0;

    // Boids global velocity matching
    float global_velocity_x = 0.0f;
    float global_velocity_y = 0.0f;

    // Total change in velocity
    float velocity_change_x = 0.f;
    float velocity_change_y = 0.f;

    const float INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("INTERACTION_RADIUS");
    const float SEPARATION_RADIUS = FLAMEGPU->environment.getProperty<float>("SEPARATION_RADIUS");
    // Iterate location messages, accumulating relevant data and counts.
    for (const auto message : FLAMEGPU->message_in.wrap(agent_x, agent_y)) {
        // Ignore self messages.
        if (message.getVariable<flamegpu::id_t>("id") != id) {
            // Get the message location and velocity.
            const float message_x = message.getVirtualX();
            const float message_y = message.getVirtualY();

            // Check interaction radius
            float separation = vec3Length(agent_x - message_x, agent_y - message_y);

            if (separation < INTERACTION_RADIUS) {
                // Update the percieved centre
                perceived_centre_x += message_x;
                perceived_centre_y += message_y;
                perceived_count++;

                // Update percieved velocity matching
                const float message_fx = message.getVariable<float>("fx");
                const float message_fy = message.getVariable<float>("fy");
                global_velocity_x += message_fx;
                global_velocity_y += message_fy;

                // Update collision centre
                if (separation < (SEPARATION_RADIUS)) {  // dependant on model size
                    // Rule 3) Avoid other nearby boids (Separation)
                    float normalizedSeparation = (separation / SEPARATION_RADIUS);
                    float invNormSep = (1.0f - normalizedSeparation);
                    float invSqSep = invNormSep * invNormSep;

                    const float collisionScale = FLAMEGPU->environment.getProperty<float>("COLLISION_SCALE");
                    velocity_change_x += collisionScale * (agent_x - message_x) * invSqSep;
                    velocity_change_y += collisionScale * (agent_y - message_y) * invSqSep;
                }
            }
        }
    }

    if (perceived_count) {
        // Divide positions/velocities by relevant counts.
        vec3Div(perceived_centre_x, perceived_centre_y, perceived_count);
        vec3Div(global_velocity_x, global_velocity_y, perceived_count);

        // Rule 1) Steer towards perceived centre of flock (Cohesion)
        float steer_velocity_x = 0.f;
        float steer_velocity_y = 0.f;

        const float STEER_SCALE = FLAMEGPU->environment.getProperty<float>("STEER_SCALE");
        steer_velocity_x = (perceived_centre_x - agent_x) * STEER_SCALE;
        steer_velocity_y = (perceived_centre_y - agent_y) * STEER_SCALE;

        velocity_change_x += steer_velocity_x;
        velocity_change_y += steer_velocity_y;

        // Rule 2) Match neighbours speeds (Alignment)
        float match_velocity_x = 0.f;
        float match_velocity_y = 0.f;

        const float MATCH_SCALE = FLAMEGPU->environment.getProperty<float>("MATCH_SCALE");
        match_velocity_x = global_velocity_x * MATCH_SCALE;
        match_velocity_y = global_velocity_y * MATCH_SCALE;

        velocity_change_x += match_velocity_x - agent_fx;
        velocity_change_y += match_velocity_y - agent_fy;
    }

    // Global scale of velocity change
    vec3Mult(velocity_change_x, velocity_change_y, FLAMEGPU->environment.getProperty<float>("GLOBAL_SCALE"));

    // Update agent velocity
    agent_fx += velocity_change_x;
    agent_fy += velocity_change_y;

    // Bound velocity
    float agent_fscale = vec3Length(agent_fx, agent_fy);
    if (agent_fscale > 1) {
        vec3Div(agent_fx, agent_fy, agent_fscale);
    }

    float minSpeed = 0.5f;
    if (agent_fscale < minSpeed) {
        // Normalise
        vec3Div(agent_fx, agent_fy, agent_fscale);

        // Scale to min
        vec3Mult(agent_fx, agent_fy, minSpeed);
    }

    // Apply the velocity
    const float TIME_SCALE = FLAMEGPU->environment.getProperty<float>("TIME_SCALE");
    agent_x += agent_fx * TIME_SCALE;
    agent_y += agent_fy * TIME_SCALE;

    // Wramp position
    const float MIN_POSITION = FLAMEGPU->environment.getProperty<float>("MIN_POSITION");
    const float MAX_POSITION = FLAMEGPU->environment.getProperty<float>("MAX_POSITION");
    wrapPosition(agent_x, agent_y, MIN_POSITION, MAX_POSITION);

    // Update global agent memory.
    FLAMEGPU->setVariable<float>("x", agent_x);
    FLAMEGPU->setVariable<float>("y", agent_y);

    FLAMEGPU->setVariable<float>("fx", agent_fx);
    FLAMEGPU->setVariable<float>("fy", agent_fy);

    return flamegpu::ALIVE;
}
"""
# outputdata agent function for Boid agents, which outputs publicly visible properties to a message list
outputdata = """
FLAMEGPU_AGENT_FUNCTION(outputdata, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    // Output each agents publicly visible properties.
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
    return flamegpu::ALIVE;
}
"""

mainTimer_start = time.monotonic()
prePopulationTimer_start = time.monotonic()
model = pyflamegpu.ModelDescription("Boids Spatial2D");

# Environment variables with default values
env = model.Environment();

# Population size to generate, if no agents are loaded from disk
env.newPropertyUInt("POPULATION_TO_GENERATE", 80000)

# Environment Bounds
env.newPropertyFloat("MIN_POSITION", 0.0)
env.newPropertyFloat("MAX_POSITION", 400.0)

# Initialisation parameter(s)
env.newPropertyFloat("INITIAL_SPEED", 1.0) # always start with a speed of 1.0

# Interaction radius
env.newPropertyFloat("INTERACTION_RADIUS", 5.0)
env.newPropertyFloat("SEPARATION_RADIUS", 1.0)

# Global Scalers
env.newPropertyFloat("TIME_SCALE", 1.0) # 1.0 for benchmarking to behave the same as the other simulators.
env.newPropertyFloat("GLOBAL_SCALE", 1.0) # 1.0 for comparing to other benchamrks

# Rule scalers
env.newPropertyFloat("STEER_SCALE", 0.03) # cohere scale?  0.03
env.newPropertyFloat("COLLISION_SCALE", 0.015) # separate_scale? 0.015
env.newPropertyFloat("MATCH_SCALE", 0.05) # match 0.05


# Define the Location 2D spatial message list
message = model.newMessageSpatial2D("location")
# Set the range and bounds.
message.setRadius(env.getPropertyFloat("INTERACTION_RADIUS"))
message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"))
message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"))

# A message to hold the location of an agent.
message.newVariableID("id")
# Spatial 2D messages implicitly have float members x and y, so they do not need to be defined
message.newVariableFloat("fx")
message.newVariableFloat("fy")
message.newVariableFloat("fz")

# Boid agent
agent = model.newAgent("Boid")
agent.newVariableFloat("x")
agent.newVariableFloat("y")
agent.newVariableFloat("fx")
agent.newVariableFloat("fy")
# Define the agents methods
outputdataDescription = agent.newRTCFunction("outputdata", outputdata)
outputdataDescription.setMessageOutput("location")
inputdataDescription = agent.newRTCFunction("inputdata", inputdata)
inputdataDescription.setMessageInput("location")

# Specify agent method dependencies, i.e. the execution order within a layer.
model.addExecutionRoot(outputdataDescription)
inputdataDescription.dependsOn(outputdataDescription)
# Build the execution graph
model.generateLayers()

# Create Model Runner
simulator = pyflamegpu.CUDASimulation(model)

# If enabled, define the visualisation for the model
if pyflamegpu.VISUALISATION:
    visualisation = simulator.getVisualisation();

    visualisation.setSimulationSpeed(10) # slow down the simulation for visualisation purposes
    env = model.Environment()
    ENV_WIDTH = env.getPropertyFloat("MAX_POSITION") - env.getPropertyFloat("MIN_POSITION")
    ENV_CENTER = env.getPropertyFloat("MIN_POSITION") + (ENV_WIDTH) / 2.0
    INIT_CAM = env.getPropertyFloat("MAX_POSITION") * 1.25
    visualisation.setInitialCameraLocation(ENV_CENTER, ENV_CENTER, INIT_CAM)
    visualisation.setInitialCameraTarget(ENV_CENTER, ENV_CENTER, 0.0)
    visualisation.setCameraSpeed(0.001 * ENV_WIDTH)
    visualisation.setViewClips(0.00001, ENV_WIDTH)
    agentVisualiser = visualisation.addAgent("Boid")
    # Position vars are named x, y so they are used by default
    agentVisualiser.setForwardXVariable("fx")
    agentVisualiser.setForwardYVariable("fy")
    agentVisualiser.setModel(pyflamegpu.STUNTPLANE)
    agentVisualiser.setModelScale(env.getPropertyFloat("SEPARATION_RADIUS")/3.0)
    # Add a settings UI
    ui = visualisation.newUIPanel("Environment")
    ui.newStaticLabel("Interaction")
    ui.newEnvironmentPropertyDragFloat("INTERACTION_RADIUS", 0.0, env.getPropertyFloat("INTERACTION_RADIUS"), 0.01) # Can't go bigger than the comms radius, which is fixed at compile time.
    ui.newEnvironmentPropertyDragFloat("SEPARATION_RADIUS", 0.0, env.getPropertyFloat("INTERACTION_RADIUS"), 0.01) # Can't go bigger than the initial interaction radius which is fixed at compile time.
    ui.newStaticLabel("Environment Scalars")
    ui.newEnvironmentPropertyDragFloat("TIME_SCALE", 0.0, 1.0, 0.0001)
    ui.newEnvironmentPropertyDragFloat("GLOBAL_SCALE", 0.0, 0.5, 0.001)
    ui.newStaticLabel("Force Scalars")
    ui.newEnvironmentPropertyDragFloat("STEER_SCALE", 0.0, 10.0, 0.001)
    ui.newEnvironmentPropertyDragFloat("COLLISION_SCALE", 0.0, 10.0, 0.001)
    ui.newEnvironmentPropertyDragFloat("MATCH_SCALE", 0.0, 10.0, 0.001)

    visualisation.activate();

# Initialisation
simulator.initialise(sys.argv)
prePopulationTimer_stop = time.monotonic()
print("pre population (s): %.6f\n"%(prePopulationTimer_stop - prePopulationTimer_start))

populationGenerationTimer_start = time.monotonic()

# If no agent states were provided, generate a population of randomly distributed agents within the environment space
if not simulator.SimulationConfig().input_file:
    env = model.Environment()
    # Uniformly distribute agents within space, with uniformly distributed initial velocity.
    # c++ random number generator engine
    rng = random.Random(simulator.SimulationConfig().random_seed)
    
    # Generate a random location within the environment bounds
    min_pos = env.getPropertyFloat("MIN_POSITION")
    max_pos = env.getPropertyFloat("MAX_POSITION")

    # Generate a random speed between 0 and the maximum initial speed
    fmagnitude = env.getPropertyFloat("INITIAL_SPEED")
    
    # Generate a population of agents, based on the relevant environment property
    populationSize = env.getPropertyUInt("POPULATION_TO_GENERATE")
    population = pyflamegpu.AgentVector(model.Agent("Boid"), populationSize)
    for i in range(populationSize):
        instance = population[i]

        # Agent position in space
        instance.setVariableFloat("x", rng.uniform(min_pos, max_pos));
        instance.setVariableFloat("y", rng.uniform(min_pos, max_pos));

        # Generate a random velocity direction
        fx = rng.uniform(-1, 1)
        fy = rng.uniform(-1, 1)
        # Use the random speed for the velocity.
        fx /= np.linalg.norm([fx, fy])
        fy /= np.linalg.norm([fx, fy])
        fx *= fmagnitude
        fy *= fmagnitude

        # Set these for the agent.
        instance.setVariableFloat("fx", fx)
        instance.setVariableFloat("fy", fy)

    simulator.setPopulationData(population);

populationGenerationTimer_stop = time.monotonic()
print("population generation (s): %.6f"%(populationGenerationTimer_stop - populationGenerationTimer_start))

# Execute the simulation
simulator.simulate()

# Print the exeuction time to stdout
print("simulate (s): %.6f"%simulator.getElapsedTimeSimulation())
print("rtc (s): %.6f"%simulator.getElapsedTimeRTCInitialisation())

# Join the visualsition if required
if pyflamegpu.VISUALISATION:
    visualisation.join()

mainTimer_stop = time.monotonic()
print("main (s): %.6f\n"%(mainTimer_stop - mainTimer_start))

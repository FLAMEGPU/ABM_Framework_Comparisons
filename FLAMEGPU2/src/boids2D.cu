#include <cstdio>
#include <cstdlib>

#include "flamegpu/flamegpu.h"

/**
 * FLAME GPU 2 implementation of the Boids flocking model in 2D, using spatial2D messaging.
 * This is based on the FLAME GPU 1 implementation, but with dynamic generation of agents. 
 * Agents are also clamped to be within the environment bounds, rather than wrapped as in FLAME GPU 1.
 */

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
 * outputdata agent function for Boid agents, which outputs publicly visible properties to a message list
 */
FLAMEGPU_AGENT_FUNCTION(outputdata, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    // Output each agents publicly visible properties.
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
    return flamegpu::ALIVE;
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
    for (const auto &message : FLAMEGPU->message_in.wrap(agent_x, agent_y)) {
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

int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("Boids Spatial3D");

    // Environment variables with default values
    flamegpu::EnvironmentDescription &env = model.Environment();

    // Population size to generate, if no agents are loaded from disk
    env.newProperty("POPULATION_TO_GENERATE", 300u);

    // Environment Bounds
    env.newProperty("MIN_POSITION", -0.5f);
    env.newProperty("MAX_POSITION", +0.5f);

    // Initialisation parameter(s)
    env.newProperty("MAX_INITIAL_SPEED", 1.0f);
    env.newProperty("MIN_INITIAL_SPEED", 0.1f);

    // Interaction radius
    env.newProperty("INTERACTION_RADIUS", 0.05f);
    env.newProperty("SEPARATION_RADIUS", 0.01f);

    // Global Scalers
    env.newProperty("TIME_SCALE", 0.0005f);
    env.newProperty("GLOBAL_SCALE", 0.15f);

    // Rule scalers
    env.newProperty("STEER_SCALE", 0.055f);
    env.newProperty("COLLISION_SCALE", 10.0f);
    env.newProperty("MATCH_SCALE", 0.015f);


    // Define the Location 2D spatial message list
    flamegpu::MessageSpatial2D::Description &message = model.newMessage<flamegpu::MessageSpatial2D>("location");
    // Set the range and bounds.
    message.setRadius(env.getProperty<float>("INTERACTION_RADIUS"));
    message.setMin(env.getProperty<float>("MIN_POSITION"), env.getProperty<float>("MIN_POSITION"));
    message.setMax(env.getProperty<float>("MAX_POSITION"), env.getProperty<float>("MAX_POSITION"));

    // A message to hold the location of an agent.
    message.newVariable<flamegpu::id_t>("id");
    // Spatial 2D messages implicitly have float members x and y, so they do not need to be defined
    message.newVariable<float>("fx");
    message.newVariable<float>("fy");
    message.newVariable<float>("fz");

    // Boid agent
    flamegpu::AgentDescription &agent = model.newAgent("Boid");
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    agent.newVariable<float>("fx");
    agent.newVariable<float>("fy");
    // Define the agents methods
    flamegpu::AgentFunctionDescription& outputdataDescription = agent.newFunction("outputdata", outputdata);
    outputdataDescription.setMessageOutput("location");
    flamegpu::AgentFunctionDescription& inputdataDescription = agent.newFunction("inputdata", inputdata);
    inputdataDescription.setMessageInput("location");

    // Specify agent method dependencies, i.e. the exeuction order within a layer.
    model.addExecutionRoot(outputdataDescription);
    inputdataDescription.dependsOn(outputdataDescription);
    // Build the execution graph
    model.generateLayers();

    // Create Model Runner
    flamegpu::CUDASimulation simulator(model);

    // If enabled, define the visualsiation for the model
#ifdef VISUALISATION
    flamegpu::visualiser::ModelVis &visualisation = simulator.getVisualisation();
    {
        flamegpu::EnvironmentDescription &env = model.Environment();
        const float ENV_WIDTH = env.getProperty<float>("MAX_POSITION") - env.getProperty<float>("MIN_POSITION");
        const float ENV_CENTER = env.getProperty<float>("MIN_POSITION") + (ENV_WIDTH) / 2.0f;
        const float INIT_CAM_DISTANCE = 1.05f;
        const float INIT_CAM = env.getProperty<float>("MAX_POSITION") * 1.25f;
        visualisation.setInitialCameraLocation(ENV_CENTER, ENV_CENTER, INIT_CAM_DISTANCE);
        visualisation.setInitialCameraTarget(ENV_CENTER, ENV_CENTER, 0.0f);
        visualisation.setCameraSpeed(0.001f * ENV_WIDTH);
        visualisation.setViewClips(0.00001f, 50);
        auto &circ_agt = visualisation.addAgent("Boid");
        // Position vars are named x, y so they are used by default
        circ_agt.setForwardXVariable("fx");
        circ_agt.setForwardYVariable("fy");
        circ_agt.setModel(flamegpu::visualiser::Stock::Models::STUNTPLANE);
        circ_agt.setModelScale(env.getProperty<float>("SEPARATION_RADIUS")/3.0f);
        // Add a settings UI
        flamegpu::visualiser::PanelVis ui = visualisation.newUIPanel("Environment");
        ui.newStaticLabel("Interaction");
        ui.newEnvironmentPropertyDrag<float>("INTERACTION_RADIUS", 0.0f, 0.05f, 0.001f);
        ui.newEnvironmentPropertyDrag<float>("SEPARATION_RADIUS", 0.0f, 0.05f, 0.001f);
        ui.newStaticLabel("Environment Scalars");
        ui.newEnvironmentPropertyDrag<float>("TIME_SCALE", 0.0f, 1.0f, 0.0001f);
        ui.newEnvironmentPropertyDrag<float>("GLOBAL_SCALE", 0.0f, 0.5f, 0.001f);
        ui.newStaticLabel("Force Scalars");
        ui.newEnvironmentPropertyDrag<float>("STEER_SCALE", 0.0f, 10.0f, 0.001f);
        ui.newEnvironmentPropertyDrag<float>("COLLISION_SCALE", 0.0f, 10.0f, 0.001f);
        ui.newEnvironmentPropertyDrag<float>("MATCH_SCALE", 0.0f, 10.0f, 0.001f);
    }
    visualisation.activate();
#endif

    // Initialisation
    simulator.initialise(argc, argv);

    // If no agent states were provided, generate a population of randomly distributed agents within the environment space
    if (simulator.getSimulationConfig().input_file.empty()) {
        flamegpu::EnvironmentDescription &env = model.Environment();
        // Uniformly distribute agents within space, with uniformly distributed initial velocity.
        // c++ random number generator engine
        std::mt19937_64 rngEngine(simulator.getSimulationConfig().random_seed);
        // Uniform distribution for agent position components
        std::uniform_real_distribution<float> position_distribution(env.getProperty<float>("MIN_POSITION"), env.getProperty<float>("MAX_POSITION"));
        // Uniform distribution of velocity direction components
        std::uniform_real_distribution<float> velocity_distribution(-1, 1);
        // Uniform distribution of velocity magnitudes
        std::uniform_real_distribution<float> velocity_magnitude_distribution(env.getProperty<float>("MIN_INITIAL_SPEED"), env.getProperty<float>("MAX_INITIAL_SPEED"));

        // Generate a population of agents, based on the relevant environment property
        const unsigned int populationSize = env.getProperty<unsigned int>("POPULATION_TO_GENERATE");
        flamegpu::AgentVector population(model.Agent("Boid"), populationSize);
        for (unsigned int i = 0; i < populationSize; i++) {
            flamegpu::AgentVector::Agent instance = population[i];

            // Agent position in space
            instance.setVariable<float>("x", position_distribution(rngEngine));
            instance.setVariable<float>("y", position_distribution(rngEngine));

            // Generate a random velocity direction
            float fx = velocity_distribution(rngEngine);
            float fy = velocity_distribution(rngEngine);
            // Generate a random speed between 0 and the maximum initial speed
            float fmagnitude = velocity_magnitude_distribution(rngEngine);
            // Use the random speed for the velocity.
            vec3Normalize(fx, fy);
            vec3Mult(fx, fy, fmagnitude);

            // Set these for the agent.
            instance.setVariable<float>("fx", fx);
            instance.setVariable<float>("fy", fy);
        }
        simulator.setPopulationData(population);
    }

    // Execute the simulation
    simulator.simulate();

    // Print the exeuction time to stdout
    fprintf(stdout, "Elapsed (s): %.6f\n", simulator.getElapsedTimeSimulation());

    // Join the visualsition if required
#ifdef VISUALISATION
    visualisation.join();
#endif
    return EXIT_SUCCESS;
}

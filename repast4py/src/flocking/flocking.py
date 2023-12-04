from repast4py import core, space, schedule, logging, random
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType
from repast4py.geometry import find_2d_nghs_periodic

from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

import numpy as np
from typing import Final, Dict, Tuple
from mpi4py import MPI

import math, sys, csv, time


mainTimer_start = time.monotonic()
prePopulationTimer_start = time.monotonic()

# Environment Bounds
MIN_POSITION: Final = 0.0
MAX_POSITION: Final = 400.0

# Initialisation parameter(s)
INITIAL_SPEED: Final = 1.0

# Interaction radius
INTERACTION_RADIUS: Final = 5.0
SEPARATION_RADIUS: Final = 1.0

# Global Scalers
TIME_SCALE: Final = 1.0 # 1.0 for benchmarking to behave the same as the other simulators.
GLOBAL_SCALE: Final = 1.0 # 1.0 for comparing to other benchmarks

# Rule scalers
STEER_SCALE: Final = 0.03 # cohere scale?  0.03
COLLISION_SCALE: Final = 0.015 # separate_scale? 0.015
MATCH_SCALE: Final = 0.05 # match 0.05

class Boid(core.Agent): 

    TYPE = 0

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Boid.TYPE, rank=rank) 
        self.xy = np.array((0.0, 0.0))
        self.fxy = np.array((0.0, 0.0))
        
    def save(self) -> Tuple:
        """Saves the state of this Boid as a Tuple.

        Used to move this Boid from one MPI rank to another.

        Returns:
            The saved state of this Boid.
        """
        return (self.uid, self.xy, self.fxy)

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = np.transpose(find_2d_nghs_periodic(np.array((pt.x, pt.y)), model.grid_box))

        # Agent position
        agent_xy = self.xy

        # Boids perceived center
        perceived_centre_xy = np.array((0.0, 0.0))
        perceived_count = 0
        
        # Boids global velocity matching
        global_velocity_xy = np.array((0.0, 0.0))
        
        # Total change in velocity
        velocity_change_xy = np.array((0.0, 0.0))

        maximum = [[], -(sys.maxsize - 1)]
        # Iterate moore neighbourhood of grid
        for ngh in nghs:
            at = dpt(ngh[0], ngh[1]) # at._reset_from_array(ngh) does not work correctly???
            # Iterate agents within current grid cell
            for obj in grid.get_agents(at):
                # Ignore self messages.
                if obj.uid[0] != self.uid[0]:
                    # Get the message location
                    message_xy = obj.xy
                    
                    # Convert message location to virtual coordinates to account for wrapping
                    xy21 = message_xy - agent_xy;
                    message_xy = np.where(abs(xy21) > MAX_POSITION / 2, message_xy - (xy21 / abs(xy21) * MAX_POSITION), message_xy)
                    
                    # Check interaction radius
                    separation = np.linalg.norm(agent_xy - message_xy)
                    
                    if separation < INTERACTION_RADIUS:
                        # Update the perceived centre
                        perceived_centre_xy += message_xy
                        perceived_count+=1
                        
                        # Update perceived velocity matching
                        global_velocity_xy += obj.fxy;                        
                        
                        # Update collision centre
                        if separation < SEPARATION_RADIUS:  # dependant on model size
                            # Rule 3) Avoid other nearby boids (Separation)
                            normalizedSeparation = (separation / SEPARATION_RADIUS)
                            invNormSep = (1.0 - normalizedSeparation)
                            invSqSep = invNormSep * invNormSep

                            velocity_change_xy += COLLISION_SCALE * (agent_xy - message_xy) * invSqSep

        if perceived_count:
            # Divide positions/velocities by relevant counts.
            perceived_centre_xy /= perceived_count
            global_velocity_xy /= perceived_count

            # Rule 1) Steer towards perceived centre of flock (Cohesion)
            steer_velocity_xy = (perceived_centre_xy - agent_xy) * STEER_SCALE

            velocity_change_xy += steer_velocity_xy

            # Rule 2) Match neighbours speeds (Alignment)
            match_velocity_xy = global_velocity_xy * MATCH_SCALE
            
            velocity_change_xy += match_velocity_xy - self.fxy

        # Global scale of velocity change
        velocity_change_xy *= GLOBAL_SCALE

        # Update agent velocity
        self.fxy += velocity_change_xy

        # Bound velocity
        if not (self.fxy[0] == 0 and self.fxy[1] == 0):
            agent_fscale = np.linalg.norm(self.fxy)
            if agent_fscale > 1:
                self.fxy /=  agent_fscale

            minSpeed = 0.5
            if agent_fscale < minSpeed:
                # Normalise
                self.fxy /= agent_fscale

                # Scale to min
                self.fxy *= minSpeed


        # Apply the velocity
        agent_xy += self.fxy * TIME_SCALE

        # Wrap position
        width = MAX_POSITION-MIN_POSITION
        for i in range(len(agent_xy)):
            if agent_xy[i] < MIN_POSITION :
                agent_xy[i] += width
            elif agent_xy[i] > MAX_POSITION :
                agent_xy[i] -= width

        # Move agent
        model.move(self, agent_xy)



agent_cache = {}


def restore_agent(agent_data: Tuple):
    """Creates an agent from the specified agent_data.

    This is used to re-create agents when they have moved from one MPI rank to another.
    The tuple returned by the agent's save() method is moved between ranks, and restore_agent
    is called for each tuple in order to create the agent on that rank. Here we also use
    a cache to cache any agents already created on this rank, and only update their state
    rather than creating from scratch.

    Args:
        agent_data: the data to create the agent from. This is the tuple returned from the agent's save() method
                    where the first element is the agent id tuple, and any remaining arguments encapsulate
                    agent state.
    """
    uid = agent_data[0]
    if uid in agent_cache:
        h = agent_cache[uid]
    else:
        h = Boid(uid[0], uid[2])
        agent_cache[uid] = h

    # restore the agent state from the agent_data tuple
    h.xy = agent_data[1]
    h.fxy = agent_data[2]
    return h
    
    
class Model:

    def __init__(self, comm, params):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()
        self.csv_log = params['csv.log']

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        grid_box = space.BoundingBox(int(MIN_POSITION), int(math.floor(MAX_POSITION/INTERACTION_RADIUS)), int(MIN_POSITION), int(math.floor(MAX_POSITION/INTERACTION_RADIUS)), 0, 0)
        box = space.BoundingBox(int(MIN_POSITION), int(MAX_POSITION), int(MIN_POSITION), int(MAX_POSITION), 0, 0)
        self.grid = space.SharedGrid('grid', bounds=grid_box, borders=BorderType.Periodic, occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.grid_box = np.array((grid_box.xmin, grid_box.xmin + grid_box.xextent - 1, grid_box.ymin, grid_box.ymin + grid_box.yextent - 1))
        self.context.add_projection(self.grid)
        
        prePopulationTimer_stop = time.monotonic()
        if self.rank == 0:
            print("pre population (s): %.6f"%(prePopulationTimer_stop - prePopulationTimer_start))
        
        populationGenerationTimer_start = time.monotonic()

        # Only rank zero generates agents, for simplicity/to avoid RNG conflict
        if self.rank == 0:
            for i in range(int(params['boid.count'])):
                h = Boid(i, self.rank)
                self.context.add(h)
                x = random.default_rng.uniform(MIN_POSITION, MAX_POSITION)
                y = random.default_rng.uniform(MIN_POSITION, MAX_POSITION)
                self.move(h, np.array((x, y)))
                fxy = np.array((random.default_rng.uniform(-1, 1), random.default_rng.uniform(-1, 1)))
                h.fxy = INITIAL_SPEED * fxy / np.linalg.norm(fxy) 

        populationGenerationTimer_stop = time.monotonic()
        if self.rank == 0:
            print("population generation (s): %.6f"%(populationGenerationTimer_stop - populationGenerationTimer_start))


    def at_end(self):
        if self.csv_log:
            # Log final agent positions to file
            self.csv_log = self.csv_log.replace(".csv", "%d.csv"%(self.comm.rank))
            with open(self.csv_log, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['"x"', '"y"', '"fx"', '"fy"'])
                for b in self.context.agents(Boid.TYPE):
                    writer.writerow([b.xy[0], b.xy[1], b.fxy[0], b.fxy[1]])

    def move(self, agent, xy):
        agent.xy = xy
        # timer.start_timer('grid_move')
        self.grid.move(agent, dpt(int(math.floor(xy[0]/INTERACTION_RADIUS)), int(math.floor(xy[1]/INTERACTION_RADIUS))))
        # timer.stop_timer('grid_move')

    def step(self):
        # print("{}: {}".format(self.rank, len(self.context.local_agents)))
        tick = self.runner.schedule.tick
        self.context.synchronize(restore_agent)

        # timer.start_timer('b_step')
        for b in self.context.agents(Boid.TYPE):
            b.step()
        # timer.stop_timer('b_step')

    def run(self):
        simulateTimer_start = time.monotonic()
        self.runner.execute()
        simulateTimer_stop = time.monotonic()
        if self.rank == 0:
            print("simulate (s): %.6f"%(simulateTimer_stop - simulateTimer_start))

    def remove_agent(self, agent):
        self.context.remove(agent)

def run(params: Dict):
    """Creates and runs the Flocking Model.

    Args:
        params: the model input parameters
    """
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()
    mainTimer_stop = time.monotonic()
    if model.rank == 0:
        print("main (s): %.6f\n"%(mainTimer_stop - mainTimer_start))


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)    

from repast4py import core, space, schedule, logging, random
from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType
from repast4py.geometry import find_2d_nghs_periodic

from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

import numpy as np
from typing import Final, Dict, Tuple
from mpi4py import MPI

import math, sys, csv, time

from functools import reduce

mainTimer_start = time.monotonic()
prePopulationTimer_start = time.monotonic()

# Configurable properties
THRESHOLD: Final = 0.375 # 0.375 == 3/8s to match integer models

A: Final = 0
B: Final = 1
UNOCCUPIED: Final = 2


class Agent(core.Agent): 

    TYPE = 0

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Agent.TYPE, rank=rank)
        self.current_type = UNOCCUPIED
        self.next_type = UNOCCUPIED
        self.happy = False
        self.available = False
        self.movement_resolved = False
        
    def save(self) -> Tuple:
        """Saves the state of this Boid as a Tuple.

        Used to move this Boid from one MPI rank to another.

        Returns:
            The saved state of this Boid.
        """
        return (self.uid, self.current_type, self.next_type, self.happy, self.available, self.movement_resolved)
        
    def determineStatus(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = np.transpose(find_2d_nghs_periodic(np.array((pt.x, pt.y)), model.grid_box))

        same_type_neighbours = 0
        diff_type_neighbours = 0

        # Iterate 3x3 Moore neighbourhood (but skip ourself)
        for ngh in nghs:
            at = dpt(ngh[0], ngh[1]) # at._reset_from_array(ngh) does not work correctly???
            if pt == at:
                continue
            # Iterate agents within current grid cell
            for obj in grid.get_agents(at):
                same_type_neighbours += int(self.current_type == obj.current_type)
                diff_type_neighbours += int(self.current_type != obj.current_type and obj.current_type != UNOCCUPIED)
        
        self.isHappy = (float(same_type_neighbours) / (same_type_neighbours + diff_type_neighbours)) > THRESHOLD if same_type_neighbours else False
        self.next_type = self.current_type if ((self.current_type != UNOCCUPIED) and self.isHappy) else UNOCCUPIED
        self.movement_resolved = (self.current_type == UNOCCUPIED) or self.isHappy
        self.available = (self.current_type == UNOCCUPIED) or not self.isHappy
        
    def bid(self, available_locations):
        if self.movement_resolved:
            return
        
        # Select a random location
        selected_location = random.default_rng.choice(available_locations) 

        # Bid for that location
        bid_grid = model.bidding_grid
        bid_grid.add(self)
        bid_grid.move(self, selected_location)
        
    def selectWinner(self):
        grid = model.grid
        pt = grid.get_location(self)
        
        bid_grid = model.bidding_grid
        bid_agents = [x for x in bid_grid.get_agents(pt)]
        # Random agent in the bucket wins
        if len(bid_agents):
            winner = random.default_rng.choice(bid_agents)
            self.next_type = winner.current_type # @todo: Does winner need to be a ghost agent?
            self.available = False
            # Remove losers from the bidding process
            for ba in bid_agents:
                if ba != winner:
                    bid_grid.remove(ba)

    def hasMoved(self):
        bid_grid = model.bidding_grid
        # Check if I won, update self and remove winning bid
        if bid_grid.contains(self):
            self.movement_resolved = True
            bid_grid.remove(self)
        
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
        h = Agent(uid[0], uid[2])
        agent_cache[uid] = h

    # restore the agent state from the agent_data tuple
    h.current_type      = agent_data[1]
    h.next_type         = agent_data[2]
    h.happy             = agent_data[3]
    h.available         = agent_data[4]
    h.movement_resolved = agent_data[5]
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

        GRID_WIDTH = int(params['grid.width'])
        grid_box = space.BoundingBox(int(0), int(GRID_WIDTH), int(0), int(GRID_WIDTH), 0, 0)
        self.grid_box = np.array((grid_box.xmin, grid_box.xmin + grid_box.xextent - 1, grid_box.ymin, grid_box.ymin + grid_box.yextent - 1))
        self.grid = space.SharedGrid('grid', bounds=grid_box, borders=BorderType.Sticky, occupancy=OccupancyType.Single, buffer_size=2, comm=comm)
        self.bidding_grid = space.SharedGrid('bidding grid', bounds=grid_box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple, buffer_size=1, comm=comm)

        self.context.add_projection(self.grid)
        self.context.add_projection(self.bidding_grid)

        
        prePopulationTimer_stop = time.monotonic()
        print("pre population (s): %.6f"%(prePopulationTimer_stop - prePopulationTimer_start))
        
        populationGenerationTimer_start = time.monotonic()

        # Calculate the total number of cells
        CELL_COUNT = GRID_WIDTH * GRID_WIDTH
        POPULATED_COUNT = int(params['population.count'])
        # Select POPULATED_COUNT unique random integers in the range [0, CELL_COUNT), by shuffling the iota.
        shuffledIota = [i for i in range(CELL_COUNT)]
        random.default_rng.shuffle(shuffledIota)
        for elementIdx in range(CELL_COUNT):
            # Create agent
            h = Agent(elementIdx, self.rank)
            self.context.add(h)
            # Setup it's location
            idx = shuffledIota[elementIdx]
            x = int(idx % GRID_WIDTH)
            y = int(idx / GRID_WIDTH)
            self.move(h, x, y)
            # If the elementIDX is below the populated count, generated a populated type, otherwise it defaults unoccupied
            if elementIdx < POPULATED_COUNT:
                h.current_type = A if random.default_rng.uniform() < 0.5 else B

        populationGenerationTimer_stop = time.monotonic()
        print("population generation (s): %.6f"%(populationGenerationTimer_stop - populationGenerationTimer_start))


    def at_end(self):
        if self.csv_log:
            # Log final agent positions to file
            with open(self.csv_log, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['"x"', '"y"', '"fx"', '"fy"'])
                for b in self.context.agents(Agent.TYPE):
                    agent_xy = self.grid.get_location(b)
                    t = b.current_type
                    writer.writerow([agent_xy.x, agent_xy.y, int(t==A or t==UNOCCUPIED), int(t==B or t==UNOCCUPIED)])

    def move(self, agent, x, y):
        # timer.start_timer('grid_move')
        self.grid.move(agent, dpt(x, y))
        # timer.stop_timer('grid_move')

    def step(self):
        # print("{}: {}".format(self.rank, len(self.context.local_agents)))
        tick = self.runner.schedule.tick
        self.context.synchronize(restore_agent)

        # timer.start_timer('b_step')
        for a in self.context.agents(Agent.TYPE):
            a.determineStatus()
        # Plan Movement Submodel
        # Break from submodel once all agents have resolved their movement
        while True:
            self.context.synchronize(restore_agent)
            # Available agents output their location
            available_spaces = [self.grid.get_location(a) for a in self.context.agents(Agent.TYPE) if a.available]
            
            # Unhappy agents bid for a new location
            for a in self.context.agents(Agent.TYPE):
                a.bid(available_spaces)
                
            self.context.synchronize(restore_agent)
            # Available locations check if anyone wants to move to them. If so, approve one and mark as unavailable
            # Update next type to the type of the mover
            for a in self.context.agents(Agent.TYPE):
                a.selectWinner()
            
            self.context.synchronize(restore_agent)
            # Successful movers mark themselves as resolved
            for a in self.context.agents(Agent.TYPE):
                a.hasMoved()
            
            self.context.synchronize(restore_agent)
            if not any([not a.movement_resolved for a in self.context.agents(Agent.TYPE)]):
                break

        self.context.synchronize(restore_agent)
        for a in self.context.agents(Agent.TYPE):
            a.current_type = a.next_type
        # timer.stop_timer('b_step')

    def run(self):
        simulateTimer_start = time.monotonic()
        self.runner.execute()
        simulateTimer_stop = time.monotonic()
        print("simulate (s): %.6f"%(simulateTimer_stop - simulateTimer_start))

    def remove_agent(self, agent):
        self.context.remove(agent)

def run(params: Dict):
    """Creates and runs the Schelling Model.

    Args:
        params: the model input parameters
    """
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)    
    mainTimer_stop = time.monotonic()
    print("main (s): %.6f\n"%(mainTimer_stop - mainTimer_start))
